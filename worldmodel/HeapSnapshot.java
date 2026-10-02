package agentwm;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.Objects;

/** Reversible heap checkpoint: retains object identity and hash-table layout.
 * Only used in a dedicated ScienceWorld JVM with no concurrent client calls.
 * Preserves world objects, goals, parser state, histories, RNG and object IDs.
 */
public final class HeapSnapshot {
    private final ArrayList<Slot> slots = new ArrayList<>();
    private final ArrayList<ArraySlot> arrays = new ArrayList<>();

    private static final class Slot {
        final Object target;
        final Field field;
        final Object value;
        Slot(Object target, Field field, Object value) {
            this.target = target; this.field = field; this.value = value;
        }
        boolean matches() throws IllegalAccessException {
            Object now = field.get(target);
            return field.getType().isPrimitive() ? Objects.equals(now, value) : now == value;
        }
        void restore() throws IllegalAccessException {
            if (!matches()) field.set(target, value);
        }
    }

    private static final class ArraySlot {
        final Object target, values;
        final int length;
        ArraySlot(Object target) {
            this.target = target;
            this.length = Array.getLength(target);
            values = Array.newInstance(target.getClass().getComponentType(), length);
            System.arraycopy(target, 0, values, 0, length);
        }
        boolean matches() {
            boolean primitive = target.getClass().getComponentType().isPrimitive();
            for (int i = 0; i < length; i++) {
                Object now = Array.get(target, i), old = Array.get(values, i);
                if (primitive ? !Objects.equals(now, old) : now != old) return false;
            }
            return true;
        }
        void restore() { System.arraycopy(values, 0, target, 0, length); }
    }

    private static boolean immutable(Object o) {
        Class<?> c = o.getClass();
        return o instanceof String || o instanceof Class || o instanceof ClassLoader ||
            o instanceof Thread || c.isEnum() || c == Boolean.class || c == Byte.class ||
            c == Short.class || c == Character.class || c == Integer.class ||
            c == Long.class || c == Float.class || c == Double.class;
    }

    public HeapSnapshot take(Object root) throws Exception {
        HeapSnapshot snapshot = new HeapSnapshot();
        ArrayDeque<Object> queue = new ArrayDeque<>();
        IdentityHashMap<Object, Boolean> seen = new IdentityHashMap<>();
        HashSet<Class<?>> statics = new HashSet<>();
        queue.add(root);
        ClassLoader loader = root.getClass().getClassLoader();
        // Scala Random drives bees/wandering animals. UniqueIdentifier assigns
        // IDs during lifecycle changes; rejected previews must consume neither.
        for (String name : new String[]{"scala.util.Random$", "util.UniqueIdentifier$"}) {
            queue.add(Class.forName(name, true, loader).getField("MODULE$").get(null));
        }
        while (!queue.isEmpty()) {
            Object o = queue.removeFirst();
            if (immutable(o) || seen.put(o, true) != null) continue;
            Class<?> type = o.getClass();
            if (type.isArray()) {
                snapshot.arrays.add(new ArraySlot(o));
                if (!type.getComponentType().isPrimitive()) {
                    for (int i = 0; i < Array.getLength(o); i++) {
                        Object v = Array.get(o, i); if (v != null) queue.add(v);
                    }
                }
                continue;
            }
            for (Class<?> c = type; c != null && c != Object.class; c = c.getSuperclass()) {
                boolean globals = (c.getName().startsWith("scienceworld.") ||
                    c.getName().startsWith("util.UniqueIdentifier") ||
                    c.getName().equals("java.util.Random")) && statics.add(c);
                for (Field f : c.getDeclaredFields()) {
                    boolean isStatic = Modifier.isStatic(f.getModifiers());
                    if (isStatic && !globals) continue;
                    if (isStatic && Modifier.isFinal(f.getModifiers()) && f.getType().isPrimitive()) continue;
                    f.setAccessible(true);
                    Object target = isStatic ? null : o;
                    Object value = f.get(target);
                    snapshot.slots.add(new Slot(target, f, value));
                    if (value != null && !f.getType().isPrimitive()) queue.add(value);
                }
            }
        }
        return snapshot;
    }

    public void restore() throws Exception {
        for (Slot slot : slots) slot.restore();
        for (ArraySlot array : arrays) array.restore();
        if (!matches()) throw new IllegalStateException("ScienceWorld heap rollback was incomplete");
    }

    public boolean matches() throws Exception {
        for (Slot slot : slots) if (!slot.matches()) return false;
        for (ArraySlot array : arrays) if (!array.matches()) return false;
        return true;
    }

    public int objectFields() { return slots.size(); }
    public int arrayCount() { return arrays.size(); }
}
