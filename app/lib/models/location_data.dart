class LocationData {
  final double latitude;
  final double longitude;
  final double? altitude;
  final double? accuracy;
  final double? heading;
  final double? speed;
  final DateTime timestamp;
  final String? placeId;
  final String? placeName;
  final String? address;
  final String? administrativeArea;
  final String? country;
  final int batteryLevel;
  final String? networkType;

  LocationData({
    required this.latitude,
    required this.longitude,
    this.altitude,
    this.accuracy,
    this.heading,
    this.speed,
    required this.timestamp,
    this.placeId,
    this.placeName,
    this.address,
    this.administrativeArea,
    this.country,
    required this.batteryLevel,
    this.networkType,
  });

  Map<String, dynamic> toJson() => {
    'latitude': latitude,
    'longitude': longitude,
    'altitude': altitude,
    'accuracy': accuracy,
    'heading': heading,
    'speed': speed,
    'timestamp': timestamp.toIso8601String(),
    'placeId': placeId,
    'placeName': placeName,
    'address': address,
    'administrativeArea': administrativeArea,
    'country': country,
    'batteryLevel': batteryLevel,
    'networkType': networkType,
  };
}
