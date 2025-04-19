import { LngLatBounds, type LngLatLike, type Map } from "maplibre-gl";
import type { Trip } from "~/model/trips_pb";

// Calculate initial center point for the map
export function tripInitialCenter(trip: Trip): LngLatLike {
  if (trip.points && trip.points.length > 0) {
    // Find a central point from the trip data
    let sumLat = 0;
    let sumLon = 0;
    let count = 0;

    for (const point of trip.points) {
      if (point.latlon) {
        sumLat += point.latlon.lat;
        sumLon += point.latlon.lon;
        count++;
      }
    }

    if (count > 0) {
      return [sumLon / count, sumLat / count];
    }
  }

  // Default center if no points available
  return [0, 0];
}

// Fit map to bounds of the trip points
export function fitMapToBounds(mapInstance: Map, trip: Trip) {
  if (!mapInstance || !trip.points?.length) return;

  // Create a bounds object
  const bounds = new LngLatBounds();

  // Add all points to the bounds
  for (const point of trip.points) {
    if (point.latlon) {
      bounds.extend([point.latlon.lon, point.latlon.lat]);
    }
  }

  // Only fit bounds if we have valid coordinates
  if (!bounds.isEmpty()) {
    mapInstance.fitBounds(bounds, {
      padding: 50,
      maxZoom: 14,
    });
  }
}
