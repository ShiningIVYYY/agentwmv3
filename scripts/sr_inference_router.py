#!/usr/bin/env python3
"""Local least-active-request routing across identical SR model replicas."""
import argparse
import asyncio
import contextlib
import time

from aiohttp import ClientError, ClientSession, ClientTimeout, web

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'transfer-encoding',
               'upgrade', 'host', 'content-length'}


def create_app(upstreams):
    app = web.Application(client_max_size=16 * 1024 * 1024)
    state = [dict(url=u.rstrip('/'), active=0, calls=0, errors=0, healthy=False)
             for u in upstreams]
    app['replicas'] = state

    async def health_loop():
        async def probe(replica):
            try:
                async with app['session'].get(replica['url'] + '/health',
                        timeout=ClientTimeout(total=4)) as response:
                    replica['healthy'] = response.status == 200
            except Exception:
                replica['healthy'] = False
        while True:
            await asyncio.gather(*(probe(r) for r in state))
            await asyncio.sleep(5)

    async def lifecycle(application):
        async with ClientSession(timeout=ClientTimeout(total=900),
                                 auto_decompress=False) as session:
            app['session'] = session
            task = asyncio.create_task(health_loop())
            yield
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def status(request):
        healthy = all(r['healthy'] for r in state)
        return web.json_response(dict(ready=healthy, replicas=state,
                                      updated=time.time()),
                                 status=200 if healthy else 503)

    async def forward(request):
        # Read the body before choosing a replica; increment before any await
        # so concurrent requests cannot all select the same idle replica.
        body = await request.read()
        available = [r for r in state if r['healthy']]
        if not available:
            raise web.HTTPServiceUnavailable(text='No healthy SR replica')
        replica = min(available, key=lambda r: (r['active'], r['calls']))
        replica['active'] += 1
        replica['calls'] += 1
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP_HEADERS}
        try:
            async with app['session'].request(request.method,
                    replica['url'] + request.rel_url.path_qs,
                    data=body, headers=headers) as upstream:
                response = web.StreamResponse(status=upstream.status,
                    headers={k: v for k, v in upstream.headers.items()
                             if k.lower() not in HOP_HEADERS})
                if upstream.status >= 500:
                    replica['errors'] += 1
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(65536):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ClientError, ConnectionError, asyncio.TimeoutError):
            replica['errors'] += 1
            replica['healthy'] = False
            raise web.HTTPBadGateway(text='SR replica connection failed')
        finally:
            replica['active'] -= 1

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/router_status', status)
    app.router.add_get('/health', status)
    app.router.add_route('*', '/{path:.*}', forward)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--upstream', action='append', required=True)
    args = parser.parse_args()
    web.run_app(create_app(args.upstream), host='127.0.0.1', port=args.port,
                access_log=None)
