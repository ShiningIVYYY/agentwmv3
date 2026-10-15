"""Check real concurrent HTTP forwarding and failed-replica exclusion."""
import asyncio
import json
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sr_inference_router import create_app
from run_sr_experiments import router_status_ready


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_preserve_body_and_distribute(self):
        gate = asyncio.Event()
        counts = [0, 0, 0]
        servers = []
        for index in range(3):
            app = web.Application()
            app.router.add_get('/health', lambda r: web.Response(text='ok'))

            async def inference(request, index=index):
                body = await request.read()
                counts[index] += 1
                await gate.wait()
                return web.Response(body=body, content_type='application/json')

            app.router.add_post('/v1/chat/completions', inference)
            server = TestServer(app)
            await server.start_server()
            servers.append(server)
        router = create_app([str(s.make_url('')).rstrip('/') for s in servers])
        client = TestClient(TestServer(router))
        await client.start_server()
        try:
            for _ in range(100):
                if all(r['healthy'] for r in router['replicas']):
                    break
                await asyncio.sleep(.02)
            body = json.dumps(dict(model='ee-scienceworld-sr', seed=0,
                messages=[dict(role='user', content='unchanged prompt')]))
            tasks = [asyncio.create_task(client.post('/v1/chat/completions',
                data=body, headers={'Content-Type': 'application/json'}))
                for _ in range(9)]
            for _ in range(100):
                if sum(counts) == 9:
                    break
                await asyncio.sleep(.02)
            self.assertEqual(counts, [3, 3, 3])
            gate.set()
            responses = await asyncio.gather(*tasks)
            for response in responses:
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.text(), body)
            self.assertEqual([r['active'] for r in router['replicas']], [0, 0, 0])
            await servers[0].close()
            await asyncio.sleep(5.2)
            self.assertFalse(router['replicas'][0]['healthy'])
            status = await (await client.get('/router_status')).json()
            urls = [r['url'] for r in router['replicas']]
            self.assertTrue(router_status_ready(
                dict(upstreams=urls[1:], retired_upstreams=urls[:1]), status))
            self.assertFalse(router_status_ready(dict(upstreams=urls), status))
            response = await client.post('/v1/chat/completions', data=body)
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.text(), body)
            self.assertEqual(counts[0], 3)
        finally:
            gate.set()
            await client.close()
            for server in servers:
                await server.close()


if __name__ == '__main__':
    unittest.main()
