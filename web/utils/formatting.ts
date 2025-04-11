import type { DateTime } from "~/model/common_pb";
import type { VisitEvent } from "~/model/ingest_pb";
import type { Trip } from "~/model/trips_pb";

// Format DateTime to a readable string
export const formatDateTime = (dateTime: DateTime | undefined | null) => {
  if (!dateTime) return "N/A";

  try {
    // Handle nested structure
    if (dateTime.year) {
      const date = new Date(
        dateTime.year,
        dateTime.month - 1,
        dateTime.day,
        dateTime.hours,
        dateTime.minutes,
        dateTime.seconds,
        dateTime.nanos / 1000000
      );
      return date.toLocaleString();
    }

    return "N/A";
  } catch (e) {
    console.error("Error parsing date:", e);
    return "Invalid date";
  }
};

// Format DateTime to a readable date only
export const formatDate = (dateTime: DateTime | undefined | null) => {
  if (!dateTime) return "N/A";

  try {
    // Handle nested structure
    if (dateTime.year) {
      const date = new Date(
        dateTime.year,
        dateTime.month - 1,
        dateTime.day,
        dateTime.hours,
        dateTime.minutes,
        dateTime.seconds,
        dateTime.nanos / 1000000
      );
      return date.toLocaleDateString();
    }

    return "N/A";
  } catch (e) {
    console.error("Error parsing date:", e);
    return "Invalid date";
  }
};

// Format trip duration
export const formatDuration = (trip: Trip) => {
  if (!trip.startTime || !trip.endTime) return "";

  try {
    const startDate = new Date(
      trip.startTime.year,
      trip.startTime.month - 1,
      trip.startTime.day,
      trip.startTime.hours,
      trip.startTime.minutes,
      trip.startTime.seconds
    );

    const endDate = new Date(
      trip.endTime.year,
      trip.endTime.month - 1,
      trip.endTime.day,
      trip.endTime.hours,
      trip.endTime.minutes,
      trip.endTime.seconds
    );

    const durationMs = endDate.getTime() - startDate.getTime();

    // Convert to hours and minutes
    const hours = Math.floor(durationMs / (1000 * 60 * 60));
    const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes} min`;
  } catch (e) {
    console.error("Error calculating duration:", e);
    return "";
  }
};

// Format time from DateTime
export const formatTime = (dateTime: DateTime | undefined) => {
  if (!dateTime) return "";

  try {
    // Handle nested structure
    if (dateTime.year) {
      const date = new Date(
        dateTime.year,
        dateTime.month - 1,
        dateTime.day,
        dateTime.hours,
        dateTime.minutes,
        dateTime.seconds,
        dateTime.nanos / 1000000
      );
      return date.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      });
    }

    return "";
  } catch (e) {
    console.error("Error parsing time:", e);
    return "";
  }
};

// Format distance in kilometers or meters
export const formatDistance = (meters: number) => {
  if (!meters && meters !== 0) return "";

  if (meters >= 1000) {
    return `${(meters / 1000).toFixed(1)} km`;
  }
  return `${Math.round(meters)} m`;
};

// Format visit duration
export const formatDurationFromVisit = (visit: VisitEvent) => {
  if (!visit.arrival || !visit.departure) return "";

  try {
    const arrivalDate = new Date(
      visit.arrival.year,
      visit.arrival.month - 1,
      visit.arrival.day,
      visit.arrival.hours,
      visit.arrival.minutes,
      visit.arrival.seconds
    );

    const departureDate = new Date(
      visit.departure.year,
      visit.departure.month - 1,
      visit.departure.day,
      visit.departure.hours,
      visit.departure.minutes,
      visit.departure.seconds
    );

    const durationMs = departureDate.getTime() - arrivalDate.getTime();

    // Convert to hours and minutes
    const hours = Math.floor(durationMs / (1000 * 60 * 60));
    const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));

    if (hours > 0) {
      return `Duration: ${hours}h ${minutes}m`;
    }
    return `Duration: ${minutes} min`;
  } catch (e) {
    console.error("Error calculating visit duration:", e);
    return "";
  }
};
