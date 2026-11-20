"""Writes the glTF test fixtures of humanart-io (dev-time only, Python standard library) [DEP-04].

Four small files, laid out here independently of the Rust reader and writer:

* triangle.gltf      - embedded data URI buffer, u16 indices, PBR material with typed KHR_materials_*
                       extensions, a data URI PNG, a sampler and KHR_texture_transform.
* sparse_morph.gltf  - external buffer "sparse morph.bin" (percent-encoded URI), u8 indices, a
                       position accessor with sparse substitution, a dense and a sparse-only morph
                       target, target names and a weights animation.
* skinned.glb        - GLB with an interleaved vertex buffer (byteStride 32, normalized u8 weights,
                       u8 joints), a skin, a node hierarchy with a matrix node, a camera, a spot light,
                       an unknown extension and extras, STEP translation and CUBICSPLINE rotation with
                       normalized i16 output, a PNG in a buffer view.
* quantized.gltf     - KHR_mesh_quantization (required): i16 positions, normalized i8 normals and u16
                       texcoords interleaved with byteStride 16, non-indexed.

Each `<name>.expect.json` maps accessor indices to the values a reader must decode (normalized
integers converted as the glTF specification says, sparse substitution applied).

Run from this directory: python -I make_fixtures.py
"""
import base64
import json
import struct

PNG_SMALL = open("../png/rgba8_i_1x1.png", "rb").read()
PNG_RGB = open("../png/c2_d8.png", "rb").read()


class Bin:
    """One buffer and its buffer views, every view starting on a 4-byte boundary."""

    def __init__(self):
        self.data = bytearray()
        self.views = []

    def view(self, raw, stride=None, target=None):
        while len(self.data) % 4:
            self.data.append(0)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if stride is not None:
            view["byteStride"] = stride
        if target is not None:
            view["target"] = target
        self.data += raw
        self.views.append(view)
        return len(self.views) - 1

    def padded(self):
        while len(self.data) % 4:
            self.data.append(0)
        return bytes(self.data)


def f32s(values):
    return struct.pack("<%df" % len(values), *values)


def flat(rows):
    return [c for row in rows for c in row]


def bounds(rows):
    return [min(c) for c in zip(*rows)], [max(c) for c in zip(*rows)]


def f32(v):
    """The value rounded to f32, as the reader will return it."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def write_json(name, doc):
    with open(name, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def triangle():
    b = Bin()
    indices = [0, 1, 2]
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    vi = b.view(struct.pack("<3H", *indices), target=34963)
    vp = b.view(f32s(flat(positions)), target=34962)
    vt = b.view(f32s(flat(uvs)), target=34962)
    pmin, pmax = bounds(positions)
    data = b.padded()
    doc = {
        "asset": {"version": "2.0", "generator": "make_fixtures.py"},
        "extensionsUsed": [
            "KHR_materials_emissive_strength",
            "KHR_materials_ior",
            "KHR_materials_specular",
            "KHR_texture_transform",
        ],
        "scene": 0,
        "scenes": [{"name": "scene", "nodes": [0]}],
        "nodes": [{"name": "tri", "mesh": 0, "translation": [0.0, 1.5, 0.0], "rotation": [0.0, 0.0, 0.6, 0.8]}],
        "meshes": [
            {
                "name": "triangle",
                "primitives": [{"attributes": {"POSITION": 1, "TEXCOORD_0": 2}, "indices": 0, "material": 0}],
            }
        ],
        "materials": [
            {
                "name": "skin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.6, 0.5, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.5,
                    "baseColorTexture": {
                        "index": 0,
                        "extensions": {"KHR_texture_transform": {"offset": [0.5, 0.0], "scale": [2.0, 2.0]}},
                    },
                },
                "doubleSided": True,
                "extensions": {
                    "KHR_materials_emissive_strength": {"emissiveStrength": 2.0},
                    "KHR_materials_ior": {"ior": 1.4},
                    "KHR_materials_specular": {"specularFactor": 0.5},
                },
                "extras": {"humanart": {"subsurfaceRadius": [1.0, 0.2, 0.1]}},
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"uri": "data:image/png;base64," + base64.b64encode(PNG_SMALL).decode()}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33648}],
        "accessors": [
            {"bufferView": vi, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": vp, "componentType": 5126, "count": 3, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": vt, "componentType": 5126, "count": 3, "type": "VEC2"},
        ],
        "bufferViews": b.views,
        "buffers": [
            {"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}
        ],
        "extras": {"humanart": {"regions": [1, 2]}},
    }
    write_json("triangle.gltf", doc)
    write_json("triangle.expect.json", {"0": indices, "1": flat(positions), "2": flat(uvs)})


def sparse_morph():
    b = Bin()
    base = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    substituted = {1: (2.0, 0.0, 0.0), 3: (0.0, 2.0, 0.0)}
    final = [substituted.get(i, p) for i, p in enumerate(base)]
    indices = [0, 1, 2, 0, 2, 3]
    target0 = [(0.0, 0.0, 0.1 * i) for i in range(4)]
    target1 = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0)]
    times = [0.0, 1.0]
    weights = [0.0, 1.0, 1.0, 0.0]
    vi = b.view(bytes(indices), target=34963)
    vp = b.view(f32s(flat(base)), target=34962)
    vsi = b.view(struct.pack("<2H", 1, 3))
    vsv = b.view(f32s(flat(substituted[k] for k in (1, 3))))
    vt0 = b.view(f32s(flat(target0)), target=34962)
    vt1i = b.view(struct.pack("<I", 2))
    vt1v = b.view(f32s([0.5, 0.5, 0.0]))
    vtime = b.view(f32s(times))
    vw = b.view(f32s(weights))
    data = b.padded()
    with open("sparse morph.bin", "wb") as f:
        f.write(data)
    fmin, fmax = bounds(final)
    t0min, t0max = bounds([tuple(f32(c) for c in r) for r in target0])
    t1min, t1max = bounds(target1)
    doc = {
        "asset": {"version": "2.0", "generator": "make_fixtures.py"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 1},
                        "indices": 0,
                        "targets": [{"POSITION": 2}, {"POSITION": 3}],
                    }
                ],
                "weights": [0.25, 0.75],
                "extras": {"targetNames": ["up", "corner"]},
            }
        ],
        "animations": [
            {
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
                "samplers": [{"input": 4, "output": 5}],
            }
        ],
        "accessors": [
            {"bufferView": vi, "componentType": 5121, "count": 6, "type": "SCALAR"},
            {
                "bufferView": vp,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": fmin,
                "max": fmax,
                "sparse": {
                    "count": 2,
                    "indices": {"bufferView": vsi, "componentType": 5123},
                    "values": {"bufferView": vsv},
                },
            },
            {"bufferView": vt0, "componentType": 5126, "count": 4, "type": "VEC3", "min": t0min, "max": t0max},
            {
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": t1min,
                "max": t1max,
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": vt1i, "componentType": 5125},
                    "values": {"bufferView": vt1v},
                },
            },
            {"bufferView": vtime, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
            {"bufferView": vw, "componentType": 5126, "count": 4, "type": "SCALAR"},
        ],
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(data), "uri": "sparse%20morph.bin"}],
    }
    write_json("sparse_morph.gltf", doc)
    write_json(
        "sparse_morph.expect.json",
        {"0": indices, "1": flat(final), "2": flat(target0), "3": flat(target1), "4": times, "5": weights},
    )


def skinned():
    b = Bin()
    positions = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 2.0, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 3
    weights = [(255, 0, 0, 0), (128, 127, 0, 0), (0, 255, 0, 0)]
    joints = [(0, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)]
    vertices = b"".join(
        f32s(p) + f32s(n) + bytes(w) + bytes(j) for p, n, w, j in zip(positions, normals, weights, joints)
    )
    vv = b.view(vertices, stride=32, target=34962)
    ibm = [
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1],
    ]
    vibm = b.view(f32s(flat(ibm)))
    times = [0.0, 1.0]
    translations = [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0)]
    h = 0.7071067811865476
    rot_rows = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, h, 0, h), (0, 0, 0, 0)]
    rot_i16 = [round(c * 32767) for c in flat(rot_rows)]
    vtime = b.view(f32s(times))
    vtr = b.view(f32s(flat(translations)))
    vrot = b.view(struct.pack("<%dh" % len(rot_i16), *rot_i16))
    vimg = b.view(PNG_RGB)
    pmin, pmax = bounds(positions)
    data = b.padded()
    doc = {
        "asset": {"version": "2.0", "generator": "make_fixtures.py", "copyright": "HumanArt test fixture"},
        "extensionsUsed": ["KHR_lights_punctual", "EXT_humanart_test"],
        "extensions": {
            "KHR_lights_punctual": {
                "lights": [
                    {
                        "name": "key",
                        "type": "spot",
                        "color": [1.0, 0.9, 0.8],
                        "intensity": 20.0,
                        "range": 10.0,
                        "spot": {"innerConeAngle": 0.2, "outerConeAngle": 0.6},
                    }
                ]
            }
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Armature", "children": [1, 3, 4]},
            {"name": "hip", "children": [2], "translation": [0.0, 1.0, 0.0]},
            {"name": "spine", "matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.5, 0, 1]},
            {"name": "body", "mesh": 0, "skin": 0},
            {
                "name": "rig",
                "camera": 0,
                "translation": [0.0, 1.6, 3.0],
                "extensions": {"KHR_lights_punctual": {"light": 0}, "EXT_humanart_test": {"x": 1}},
                "extras": {"landmarks": {"nose_tip": [0.0, 1.62, 0.1]}},
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "WEIGHTS_0": 2, "JOINTS_0": 3},
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "skins": [{"inverseBindMatrices": 4, "skeleton": 1, "joints": [1, 2]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": vimg, "mimeType": "image/png"}],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.8, "znear": 0.01, "aspectRatio": 1.5}}],
        "animations": [
            {
                "name": "idle",
                "channels": [
                    {"sampler": 0, "target": {"node": 1, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 2, "path": "rotation"}},
                ],
                "samplers": [
                    {"input": 5, "output": 6, "interpolation": "STEP"},
                    {"input": 5, "output": 7, "interpolation": "CUBICSPLINE"},
                ],
            }
        ],
        "accessors": [
            {"bufferView": vv, "componentType": 5126, "count": 3, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": vv, "byteOffset": 12, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": vv, "byteOffset": 24, "componentType": 5121, "normalized": True, "count": 3, "type": "VEC4"},
            {"bufferView": vv, "byteOffset": 28, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": vibm, "componentType": 5126, "count": 2, "type": "MAT4"},
            {"bufferView": vtime, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
            {"bufferView": vtr, "componentType": 5126, "count": 2, "type": "VEC3"},
            {"bufferView": vrot, "componentType": 5122, "normalized": True, "count": 6, "type": "VEC4"},
        ],
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(data)}],
    }
    text = json.dumps(doc, separators=(",", ":")).encode()
    while len(text) % 4:
        text += b" "
    total = 12 + 8 + len(text) + 8 + len(data)
    glb = struct.pack("<4sII", b"glTF", 2, total)
    glb += struct.pack("<I4s", len(text), b"JSON") + text
    glb += struct.pack("<I4s", len(data), b"BIN\0") + data
    with open("skinned.glb", "wb") as f:
        f.write(glb)
    write_json(
        "skinned.expect.json",
        {
            "0": flat(positions),
            "1": flat(normals),
            "2": [c / 255 for c in flat(weights)],
            "3": flat(joints),
            "4": flat(ibm),
            "5": times,
            "6": flat(translations),
            "7": [max(c / 32767, -1.0) for c in rot_i16],
        },
    )


def quantized():
    positions = [(0, 0, 0), (1000, 0, -500), (0, 1000, 32767)]
    normals = [(0, 0, 127), (0, -128, 0), (90, 90, 0)]
    uvs = [(0, 0), (65535, 0), (0, 32768)]
    vertices = b"".join(
        struct.pack("<3h2x", *p) + struct.pack("<3bx", *n) + struct.pack("<2H", *t)
        for p, n, t in zip(positions, normals, uvs)
    )
    b = Bin()
    vv = b.view(vertices, stride=16, target=34962)
    data = b.padded()
    pmin, pmax = bounds(positions)
    doc = {
        "asset": {"version": "2.0", "generator": "make_fixtures.py"},
        "extensionsUsed": ["KHR_mesh_quantization"],
        "extensionsRequired": ["KHR_mesh_quantization"],
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "scale": [0.001, 0.001, 0.001]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}}]}],
        "accessors": [
            {"bufferView": vv, "componentType": 5122, "count": 3, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": vv, "byteOffset": 8, "componentType": 5120, "normalized": True, "count": 3, "type": "VEC3"},
            {"bufferView": vv, "byteOffset": 12, "componentType": 5123, "normalized": True, "count": 3, "type": "VEC2"},
        ],
        "bufferViews": b.views,
        "buffers": [
            {"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}
        ],
    }
    write_json("quantized.gltf", doc)
    write_json(
        "quantized.expect.json",
        {
            "0": flat(positions),
            "1": [max(c / 127, -1.0) for c in flat(normals)],
            "2": [c / 65535 for c in flat(uvs)],
        },
    )


triangle()
sparse_morph()
skinned()
quantized()
