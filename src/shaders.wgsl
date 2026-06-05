// Originally written in 2023 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

const FLT_MAX: f32 = 3.40282346638528859812e+38;
const EPSILON: f32 = 1e-3;
const TWO_PI: f32 = 6.2831853;

const MAX_PATH_LENGTH: u32 = 13u;
const GLASS_F0: vec3f = vec3(0.04);

struct Uniforms {
  camera: CameraUniforms,
  width: u32,
  height: u32,
  frame_count: u32,
  samples_per_frame: u32,
  audio_energy: f32,
  reactive_lights: f32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct CameraUniforms {
  origin: vec3f,
  fov_y: f32,
  u: vec3f,
  // TODO: focus_distance: f32,
  v: vec3f,
  w: vec3f,
}

struct Rng {
  state: u32,
}
var<private> rng: Rng;

fn init_rng(pixel: vec2u) {
  // Seed the PRNG using the scalar index of the pixel and the current frame count.
  let seed = (pixel.x + pixel.y * uniforms.width) ^ jenkins_hash(uniforms.frame_count);
  rng.state = jenkins_hash(seed);
}

// A slightly modified version of the "One-at-a-Time Hash" function by Bob Jenkins.
// See https://www.burtleburtle.net/bob/hash/doobs.html
fn jenkins_hash(i: u32) -> u32 {
  var x = i;
  x += x << 10u;
  x ^= x >> 6u;
  x += x << 3u;
  x ^= x >> 11u;
  x += x << 15u;
  return x;
}

// The 32-bit "xor" function from Marsaglia G., "Xorshift RNGs", Section 3.
fn xorshift32() -> u32 {
  var x = rng.state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng.state = x;
  return x;
}

// Returns a random float in the range [0...1]. This sets the floating point exponent to zero and
// sets the most significant 23 bits of a random 32-bit unsigned integer as the mantissa. That
// generates a number in the range [1, 1.9999999], which is then mapped to [0, 0.9999999] by
// subtraction. See Ray Tracing Gems II, Section 14.3.4.
fn rand_f32() -> f32 {
  return bitcast<f32>(0x3f800000u | (xorshift32() >> 9u)) - 1.;
}

// Convert RGB to a grayscale luminance value
fn luminance(rgb: vec3f) -> f32 {
  return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

// Uniformly sample a unit sphere centered at the origin
fn sample_sphere() -> vec3f {
  let r0 = rand_f32();
  let r1 = rand_f32();

  // Map r0 to [-1, 1]
  let y = 1. - 2. * r0;

  // Compute the projected radius on the xz-plane using Pythagorean theorem
  let xz_r = sqrt(1. - y * y);

  let phi = TWO_PI * r1;
  return vec3(xz_r * cos(phi), y, xz_r * sin(phi));
}

struct Intersection {
  normal: vec3f,
  t: f32,
  material_index: u32,
}

fn no_intersection() -> Intersection {
  return Intersection(vec3(0.), -1., 0);
}

fn is_intersection_valid(hit: Intersection) -> bool {
  return hit.t > 0.;
}

struct Object {
  data0: vec3f,
  kind: u32,
  data1: vec3f,
  material_index: u32,
  data2: vec3f,
  _pad: u32,
}

const OBJECT_KIND_SPHERE: u32 = 0u;
const OBJECT_KIND_BOX: u32 = 1u;
const OBJECT_KIND_CYLINDER: u32 = 2u;

fn intersect_sphere(ray: Ray, object: Object) -> Intersection {
  let center = object.data0;
  let radius = object.data1.x;
  let v = ray.origin - center;
  let a = dot(ray.direction, ray.direction);
  let b = dot(v, ray.direction);
  let c = dot(v, v) - radius * radius;

  let d = b * b - a * c;
  if d < 0. {
    return no_intersection();
  }

  let sqrt_d = sqrt(d);
  let recip_a = 1. / a;
  let mb = -b;
  let t1 = (mb - sqrt_d) * recip_a;
  let t2 = (mb + sqrt_d) * recip_a;
  let t = select(t2, t1, t1 >= EPSILON);
  if t < EPSILON {
    return no_intersection();
  }

  let p = point_on_ray(ray, t);
  let N = (p - center) / radius;
  return Intersection(N, t, object.material_index);
}

fn intersect_box(ray: Ray, object: Object) -> Intersection {
  let box_min = object.data0;
  let box_max = object.data1;
  var t_min = -FLT_MAX;
  var t_max = FLT_MAX;

  for (var axis = 0u; axis < 3u; axis += 1u) {
    let origin = ray.origin[axis];
    let direction = ray.direction[axis];
    let lo = box_min[axis];
    let hi = box_max[axis];
    if abs(direction) < 1e-8 {
      if origin < lo || origin > hi {
        return no_intersection();
      }
    } else {
      let inv = 1.0 / direction;
      var t0 = (lo - origin) * inv;
      var t1 = (hi - origin) * inv;
      if t0 > t1 {
        let tmp = t0;
        t0 = t1;
        t1 = tmp;
      }
      t_min = max(t_min, t0);
      t_max = min(t_max, t1);
      if t_min > t_max {
        return no_intersection();
      }
    }
  }

  let t = select(t_max, t_min, t_min >= EPSILON);
  if t < EPSILON {
    return no_intersection();
  }

  let p = point_on_ray(ray, t);
  let eps = 1e-3;
  var normal = vec3f(0.0);
  if abs(p.x - box_min.x) < eps {
    normal = vec3f(-1.0, 0.0, 0.0);
  } else if abs(p.x - box_max.x) < eps {
    normal = vec3f(1.0, 0.0, 0.0);
  } else if abs(p.y - box_min.y) < eps {
    normal = vec3f(0.0, -1.0, 0.0);
  } else if abs(p.y - box_max.y) < eps {
    normal = vec3f(0.0, 1.0, 0.0);
  } else if abs(p.z - box_min.z) < eps {
    normal = vec3f(0.0, 0.0, -1.0);
  } else {
    normal = vec3f(0.0, 0.0, 1.0);
  }
  return Intersection(normal, t, object.material_index);
}

fn intersect_cylinder(ray: Ray, object: Object) -> Intersection {
  let center = object.data0;
  let radius = object.data1.x;
  let y_min = object.data1.y;
  let y_max = object.data1.z;
  var closest_t = FLT_MAX;
  var normal = vec3f(0.0);

  let oc = ray.origin - center;
  let a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
  let b = oc.x * ray.direction.x + oc.z * ray.direction.z;
  let c = oc.x * oc.x + oc.z * oc.z - radius * radius;
  let d = b * b - a * c;
  if a > 1e-8 && d >= 0.0 {
    let sqrt_d = sqrt(d);
    let recip_a = 1.0 / a;
    let t0 = (-b - sqrt_d) * recip_a;
    let t1 = (-b + sqrt_d) * recip_a;
    for (var i = 0u; i < 2u; i += 1u) {
      let t = select(t1, t0, i == 0u);
      let y = ray.origin.y + ray.direction.y * t;
      if t >= EPSILON && y >= y_min && y <= y_max && t < closest_t {
        closest_t = t;
        let p = point_on_ray(ray, t);
        normal = normalize(vec3f(p.x - center.x, 0.0, p.z - center.z));
      }
    }
  }

  if abs(ray.direction.y) > 1e-8 {
    let cap_values = vec2f(y_min, y_max);
    for (var i = 0u; i < 2u; i += 1u) {
      let cap_y = cap_values[i];
      let t = (cap_y - ray.origin.y) / ray.direction.y;
      let p = point_on_ray(ray, t);
      let dx = p.x - center.x;
      let dz = p.z - center.z;
      if t >= EPSILON && dx * dx + dz * dz <= radius * radius && t < closest_t {
        closest_t = t;
        normal = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, -1.0, 0.0), i == 0u);
      }
    }
  }

  if closest_t == FLT_MAX {
    return no_intersection();
  }
  return Intersection(normal, closest_t, object.material_index);
}

fn intersect_object(ray: Ray, object: Object) -> Intersection {
  if object.kind == OBJECT_KIND_SPHERE {
    return intersect_sphere(ray, object);
  }
  if object.kind == OBJECT_KIND_BOX {
    return intersect_box(ray, object);
  }
  if object.kind == OBJECT_KIND_CYLINDER {
    return intersect_cylinder(ray, object);
  }
  return no_intersection();
}

// Test every sphere bucketed into `cell`, updating the running closest hit.
fn test_cell(ray: Ray, cell: u32, closest: ptr<function, Intersection>) {
  let base = cell * 2u;
  let start = cell_ranges[base];
  let count = cell_ranges[base + 1u];
  for (var k = 0u; k < count; k += 1u) {
    let si = sphere_indices[start + k];
    let hit = intersect_object(ray, objects[si]);
    if hit.t > 0. && hit.t < (*closest).t {
      *closest = hit;
    }
  }
}

// Find the closest intersection by walking the uniform grid (2D DDA over XZ),
// testing only spheres in the visited cells. A single-cell grid degenerates to
// a brute-force loop (used by static scenes).
fn intersect_scene(ray: Ray) -> Intersection {
  var closest = no_intersection();
  closest.t = FLT_MAX;

  let nx = i32(grid.nx);
  let nz = i32(grid.nz);

  if nx == 1 && nz == 1 {
    test_cell(ray, 0u, &closest);
    if closest.t < FLT_MAX { return closest; }
    return no_intersection();
  }

  let cell = 1. / grid.inv_cell;
  let min_x = grid.min_x;
  let min_z = grid.min_z;
  let max_x = min_x + f32(nx) * cell;
  let max_z = min_z + f32(nz) * cell;

  let ox = ray.origin.x;
  let oz = ray.origin.z;
  let dx = ray.direction.x;
  let dz = ray.direction.z;

  // Clip the ray to the grid's XZ bounds (slab test); y is unbounded.
  var t_enter = 0.;
  var t_exit = FLT_MAX;
  if abs(dx) < 1e-8 {
    if ox < min_x || ox > max_x { return no_intersection(); }
  } else {
    let inv = 1. / dx;
    var t0 = (min_x - ox) * inv;
    var t1 = (max_x - ox) * inv;
    if t0 > t1 { let tmp = t0; t0 = t1; t1 = tmp; }
    t_enter = max(t_enter, t0);
    t_exit = min(t_exit, t1);
  }
  if abs(dz) < 1e-8 {
    if oz < min_z || oz > max_z { return no_intersection(); }
  } else {
    let inv = 1. / dz;
    var t0 = (min_z - oz) * inv;
    var t1 = (max_z - oz) * inv;
    if t0 > t1 { let tmp = t0; t0 = t1; t1 = tmp; }
    t_enter = max(t_enter, t0);
    t_exit = min(t_exit, t1);
  }
  if t_enter > t_exit { return no_intersection(); }
  t_enter = max(t_enter, 0.);

  // Starting cell at the entry point.
  let ex = ox + dx * t_enter;
  let ez = oz + dz * t_enter;
  var ix = clamp(i32(floor((ex - min_x) * grid.inv_cell)), 0, nx - 1);
  var iz = clamp(i32(floor((ez - min_z) * grid.inv_cell)), 0, nz - 1);

  let step_x = select(-1, 1, dx >= 0.);
  let step_z = select(-1, 1, dz >= 0.);

  // Parameter to the next x/z cell boundary, and per-cell increments.
  let bx = min_x + f32(ix + select(0, 1, dx >= 0.)) * cell;
  let bz = min_z + f32(iz + select(0, 1, dz >= 0.)) * cell;
  var t_max_x = select(FLT_MAX, (bx - ox) / dx, abs(dx) >= 1e-8);
  var t_max_z = select(FLT_MAX, (bz - oz) / dz, abs(dz) >= 1e-8);
  let t_delta_x = select(FLT_MAX, cell / abs(dx), abs(dx) >= 1e-8);
  let t_delta_z = select(FLT_MAX, cell / abs(dz), abs(dz) >= 1e-8);

  loop {
    test_cell(ray, u32(iz) * grid.nx + u32(ix), &closest);

    // We have now covered everything up to t_cell_exit; a closer hit cannot
    // lie beyond it, so stop as soon as the closest hit is within reach.
    let t_cell_exit = min(t_max_x, t_max_z);
    if closest.t <= t_cell_exit { return closest; }
    if t_cell_exit > t_exit { break; }

    if t_max_x < t_max_z {
      ix += step_x;
      t_max_x += t_delta_x;
      if ix < 0 || ix >= nx { break; }
    } else {
      iz += step_z;
      t_max_z += t_delta_z;
      if iz < 0 || iz >= nz { break; }
    }
  }

  if closest.t < FLT_MAX { return closest; }
  return no_intersection();
}

struct Scatter {
  attenuation: vec3f,
  ray: Ray,
}

fn sample_lambertian(normal: vec3f) -> vec3f {
  return normal + sample_sphere() * (1. - EPSILON);
}

fn schlick_f0_from_ior(ior: f32) -> f32 {
  let sqrt_f0 = (ior - 1.) / (ior + 1.);
  return sqrt_f0 * sqrt_f0;
}

// u5 = (1 - cos_theta)^5
fn schlick_fresnel(f0: f32, u5: f32) -> f32 {
  return mix(f0, 1., u5);
}

// u5 = (1 - cos_theta)^5
fn schlick_fresnel_vec3(f0: vec3f, u5: f32) -> vec3f {
  return mix(f0, vec3(1.), u5);
}

fn scatter(input_ray: Ray, hit: Intersection, material: Material) -> Scatter {
  let incident = normalize(input_ray.direction);
  let incident_dot_normal = dot(incident, hit.normal);
  let is_front_face = incident_dot_normal < 0.;
  let N = select(-hit.normal, hit.normal, is_front_face);
  let cos_theta = abs(incident_dot_normal);

  // `ior` and `ref_ratio` only have meaning if the material is transmissive.
  let is_transmissive = material.metallic_or_ior < 0.;
  let ior = abs(material.metallic_or_ior);
  let ref_ratio = select(ior, 1. / ior, is_front_face);

  // The (1 - cos(theta))^5 term from the Schlick approximation.
  let u = 1 - cos_theta;
  let u2 = u * u;
  let u5 = u2 * u2 * u;

  // Determine whether to use specular reflection.
  var choose_specular = false;
  var attenuation = material.color;
  if is_transmissive {
    let cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
    let f0 = schlick_f0_from_ior(ref_ratio);
    choose_specular = cannot_refract || schlick_fresnel(f0, u5) > rand_f32();
    if choose_specular {
      attenuation = vec3(1.);
    }
  } else if material.metallic_or_ior > 0. {
    let metallic = material.metallic_or_ior;
    let f0 = mix(GLASS_F0, material.color, metallic);
    let F = schlick_fresnel_vec3(f0, u5);

    let specular = F;
    let diffuse = (material.color - GLASS_F0) * (1. - metallic) * (1. - u5);

    let S = luminance(specular);
    let D = luminance(diffuse);
    let specular_pdf = S / (S + D);

    choose_specular = specular_pdf > rand_f32();
    if choose_specular {
      attenuation = specular / specular_pdf;
    } else {
      attenuation = diffuse / (1. - specular_pdf);
    }
  }

  var scattered: vec3f;
  if choose_specular {
    scattered = reflect(incident, N);
  } else if is_transmissive {
    scattered = refract(incident, N, ref_ratio);
  } else {
    scattered = sample_lambertian(N);
  }
  let output_ray = Ray(point_on_ray(input_ray, hit.t), scattered);
  return Scatter(attenuation, output_ray);
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
}

fn point_on_ray(ray: Ray, t: f32) -> vec3<f32> {
  return ray.origin + t * ray.direction;
}

struct Material {
  color: vec3f,
  metallic_or_ior: f32,
  // Light emitted by the surface (radiance). Zero for non-emitters.
  emission: vec3f,
}

fn sky_color(ray: Ray) -> vec3f {
  let t = 0.5 * (normalize(ray.direction).y + 1.);
  return (1. - t) * vec3(1.) + t * vec3(0.3, 0.5, 1.);
}

// The color seen by a ray that escapes the scene: the procedural sky blended
// toward a solid background color by `sky_amount` (0 = solid, 1 = full sky).
fn background(ray: Ray) -> vec3f {
  let solid = vec3(grid.bg_r, grid.bg_g, grid.bg_b);
  return mix(solid, sky_color(ray), grid.sky_amount);
}

// Header for the uniform-grid accelerator + per-scene background (mirrors
// `GridHeader` in scene.rs).
struct GridHeader {
  min_x: f32,
  min_z: f32,
  inv_cell: f32,
  nx: u32,
  nz: u32,
  bg_r: f32,
  bg_g: f32,
  bg_b: f32,
  sky_amount: f32,
}

@group(1) @binding(0) var<storage> materials: array<Material>;
@group(1) @binding(1) var<storage> objects: array<Object>;
@group(1) @binding(2) var<storage> grid: GridHeader;
@group(1) @binding(3) var<storage> cell_ranges: array<u32>;
@group(1) @binding(4) var<storage> sphere_indices: array<u32>;

@group(0) @binding(1) var radiance_samples_old: texture_2d<f32>;
@group(0) @binding(2) var radiance_samples_new: texture_storage_2d<rgba32float, write>;

alias TriangleVertices = array<vec2f, 6>;
var<private> vertices: TriangleVertices = TriangleVertices(
  vec2f(-1.0,  1.0),
  vec2f(-1.0, -1.0),
  vec2f( 1.0,  1.0),
  vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0),
  vec2f( 1.0, -1.0),
);

@vertex fn path_tracer_vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  return vec4f(vertices[vid], 0.0, 1.0);
}

fn trace_path_sample(pos: vec2f) -> vec3f {
  let origin = uniforms.camera.origin;
  let focus_distance = 1.;
  let aspect_ratio = f32(uniforms.width) / f32(uniforms.height);
  let fov_y = uniforms.camera.fov_y;
  let viewport_height = 2. * tan(fov_y * 0.5) * focus_distance;
  let viewport_width = viewport_height * aspect_ratio;

  // Offset and normalize the viewport coordinates of the ray.
  let offset = vec2(rand_f32() - 0.5, rand_f32() - 0.5);
  var uv = (pos.xy + offset) / vec2f(f32(uniforms.width - 1u), f32(uniforms.height - 1u));

  // Map `uv` from y-down normalized coordinates to scaled NDC viewport.
  uv = (uv - vec2(0.5)) * vec2(viewport_width, -viewport_height);

  // Compute the world-space ray direction by rotating the camera-space vector into a new
  // basis.
  let camera_rotation = mat3x3(uniforms.camera.u, uniforms.camera.v, uniforms.camera.w);
  let direction = camera_rotation * vec3(uv, focus_distance);
  var ray = Ray(origin, direction);
  var throughput = vec3f(1.);
  var radiance_sample = vec3(0.);

  var path_length = 0u;
  while path_length < MAX_PATH_LENGTH {
    let hit = intersect_scene(ray);
    if !is_intersection_valid(hit) {
      // If no intersection was found, return the background and terminate.
      radiance_sample += throughput * background(ray);
      break;
    }

    let material = materials[hit.material_index];

    // Add any light emitted by the surface, scaled by the path's throughput.
    let emission_pulse = 1.0 + uniforms.audio_energy * uniforms.reactive_lights;
    radiance_sample += throughput * material.emission * emission_pulse;

    let scattered = scatter(ray, hit, material);
    throughput *= scattered.attenuation;
    ray = scattered.ray;
    path_length += 1u;
  }

  return radiance_sample;
}

@fragment fn path_tracer_fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let sample_count = max(uniforms.samples_per_frame, 1u);
  var radiance_sample = vec3(0.);
  for (var sample_index = 0u; sample_index < sample_count; sample_index += 1u) {
    init_rng(vec2u(pos.xy) + vec2u(sample_index * 741103597u, sample_index * 1597334677u));
    radiance_sample += trace_path_sample(pos.xy);
  }
  radiance_sample /= f32(sample_count);

  // Fetch the old sum of samples.
  var old_sum: vec3f;
  if uniforms.frame_count > 1 {
    old_sum = textureLoad(radiance_samples_old, vec2u(pos.xy), 0).xyz;
  } else {
    old_sum = vec3(0.);
  }

  // Compute and store the new sum.
  let new_sum = radiance_sample + old_sum;
  textureStore(radiance_samples_new, vec2u(pos.xy), vec4(new_sum, 0.));

  // Display the average after gamma correction (gamma = 2.2)
  let color = new_sum / f32(uniforms.frame_count);
  return vec4(pow(color, vec3(1. / 2.2)), 1.);
}
