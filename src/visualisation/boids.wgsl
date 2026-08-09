struct Viewport {
    size: vec2<f32>,
    padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> viewport: Viewport;

struct VertexInput {
    @location(0) local_position: vec2<f32>,
    @location(1) position: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) colour: vec4<f32>,
    @location(4) effects: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) colour: vec4<f32>,
}

fn srgb_to_linear_channel(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        return encoded / 12.92;
    }
    return pow((encoded + 0.055) / 1.055, 2.4);
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let speed = length(input.velocity);
    var direction = vec2<f32>(1.0, 0.0);
    if speed > 0.001 {
        direction = input.velocity / speed;
    }
    let perpendicular = vec2<f32>(-direction.y, direction.x);
    let visibility = clamp(input.effects.x, 0.0, 1.0);
    let eased_visibility = visibility * visibility * (3.0 - 2.0 * visibility);
    let pulse_scale = 1.0 + clamp(input.effects.y, 0.0, 1.0) * 0.6;
    let world_position = input.position
        + direction * input.local_position.x * 9.0 * eased_visibility * pulse_scale
        + perpendicular * input.local_position.y * 4.0 * eased_visibility * pulse_scale;
    let normalized = world_position / viewport.size;

    var output: VertexOutput;
    output.clip_position = vec4<f32>(
        normalized.x * 2.0 - 1.0,
        1.0 - normalized.y * 2.0,
        0.0,
        1.0,
    );
    output.colour = vec4<f32>(
        srgb_to_linear_channel(input.colour.r),
        srgb_to_linear_channel(input.colour.g),
        srgb_to_linear_channel(input.colour.b),
        input.colour.a * eased_visibility,
    );
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.colour;
}
