# Coordinate precison

In certain, undefined conditions, the geo library will reduce the precision of a projected
point on a LineString without precision type parameter built from Coord<f64> to f32,
which can result in errors in the hundreds of meters range!

Always specify LineString<f64>.
