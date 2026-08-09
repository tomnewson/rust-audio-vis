use super::*;

#[test]
fn resolution_does_not_change_logical_canvas_size_at_the_same_aspect_ratio() {
    assert_eq!(
        CanvasSize::from_surface(640, 480),
        CanvasSize::from_surface(3_840, 2_880)
    );
}

#[test]
fn four_k_widescreen_canvas_preserves_square_geometry() {
    let canvas = CanvasSize::from_surface(3_840, 2_160);

    assert!((canvas.width - 853.333_3).abs() < 0.001);
    assert_eq!(canvas.height, 480.0);
    assert!((3_840.0 / canvas.width - 2_160.0 / canvas.height).abs() < 0.001);
}

#[test]
fn zero_sized_surface_uses_the_default_canvas() {
    assert_eq!(CanvasSize::from_surface(0, 0), CanvasSize::default());
}
