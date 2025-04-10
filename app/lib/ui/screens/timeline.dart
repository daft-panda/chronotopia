// timeline_screen.dart
import 'package:chronotopia_app/services/background_service.dart';
import 'package:flutter/cupertino.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

class TimelineScreen extends StatefulWidget {
  const TimelineScreen({super.key});

  @override
  _TimelineScreenState createState() => _TimelineScreenState();
}

class _TimelineScreenState extends State<TimelineScreen> {
  // This would normally come from your gRPC calls to the backend
  final List<TripData> _trips = [
    TripData(
      id: '1',
      startTime: DateTime.now().subtract(const Duration(hours: 2)),
      endTime: DateTime.now().subtract(const Duration(hours: 1, minutes: 30)),
      startPlace: 'Home',
      endPlace: 'Office',
      activityType: 'in_vehicle',
      distance: 8.5,
      duration: 30,
    ),
    TripData(
      id: '2',
      startTime: DateTime.now().subtract(const Duration(days: 1, hours: 5)),
      endTime: DateTime.now().subtract(const Duration(days: 1, hours: 4)),
      startPlace: 'Office',
      endPlace: 'Coffee Shop',
      activityType: 'walking',
      distance: 0.8,
      duration: 12,
    ),
    TripData(
      id: '3',
      startTime: DateTime.now().subtract(const Duration(days: 1, hours: 3)),
      endTime: DateTime.now().subtract(const Duration(days: 1, hours: 2)),
      startPlace: 'Coffee Shop',
      endPlace: 'Office',
      activityType: 'walking',
      distance: 0.8,
      duration: 15,
    ),
    TripData(
      id: '4',
      startTime: DateTime.now().subtract(const Duration(days: 2, hours: 8)),
      endTime: DateTime.now().subtract(
        const Duration(days: 2, hours: 7, minutes: 30),
      ),
      startPlace: 'Home',
      endPlace: 'Gym',
      activityType: 'on_bicycle',
      distance: 3.2,
      duration: 25,
    ),
  ];

  Map<String, List<TripData>> _groupTripsByDate() {
    final Map<String, List<TripData>> grouped = {};

    for (final trip in _trips) {
      final dateKey = DateFormat('yyyy-MM-dd').format(trip.startTime);
      if (!grouped.containsKey(dateKey)) {
        grouped[dateKey] = [];
      }
      grouped[dateKey]!.add(trip);
    }

    // Sort each group by time
    for (final key in grouped.keys) {
      grouped[key]!.sort((a, b) => b.startTime.compareTo(a.startTime));
    }

    return grouped;
  }

  @override
  Widget build(BuildContext context) {
    final isTracking = Provider.of<BackgroundService>(context).isRunning;
    final groupedTrips = _groupTripsByDate();
    final sortedDates =
        groupedTrips.keys.toList()..sort((a, b) => b.compareTo(a));
    final textTheme = CupertinoTheme.of(context).textTheme;

    return CupertinoPageScaffold(
      navigationBar: const CupertinoNavigationBar(
        middle: Text('Your Timeline'),
      ),
      child: SafeArea(
        child: Column(
          children: [
            // Status banner
            Container(
              color:
                  isTracking
                      ? CupertinoColors.activeGreen.withOpacity(0.1)
                      : CupertinoColors.systemOrange.withOpacity(0.1),
              padding: const EdgeInsets.all(8.0),
              child: Row(
                children: [
                  Icon(
                    isTracking
                        ? CupertinoIcons.location_fill
                        : CupertinoIcons.location_slash,
                    color:
                        isTracking
                            ? CupertinoColors.activeGreen
                            : CupertinoColors.systemOrange,
                  ),
                  const SizedBox(width: 8.0),
                  Expanded(
                    child: Text(
                      isTracking
                          ? 'Tracking is active. Your timeline is being built.'
                          : 'Tracking is paused. Enable tracking to continue building your timeline.',
                      style: textTheme.textStyle.copyWith(
                        color:
                            isTracking
                                ? CupertinoColors.activeGreen
                                : CupertinoColors.systemOrange,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Timeline
            Expanded(
              child:
                  sortedDates.isEmpty
                      ? Center(
                        child: Text(
                          'No trips recorded yet.\nEnable tracking and move around to build your timeline.',
                          textAlign: TextAlign.center,
                          style: textTheme.textStyle,
                        ),
                      )
                      : ListView.builder(
                        itemCount: sortedDates.length,
                        itemBuilder: (context, index) {
                          final dateKey = sortedDates[index];
                          final trips = groupedTrips[dateKey]!;

                          return Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Date header
                              Padding(
                                padding: const EdgeInsets.all(16.0),
                                child: Text(
                                  _formatDateHeader(dateKey),
                                  style: textTheme.navTitleTextStyle.copyWith(
                                    color: CupertinoColors.activeBlue,
                                  ),
                                ),
                              ),

                              // Trips for this date
                              ListView.builder(
                                shrinkWrap: true,
                                physics: const NeverScrollableScrollPhysics(),
                                itemCount: trips.length,
                                itemBuilder: (context, tripIndex) {
                                  final trip = trips[tripIndex];
                                  return _buildTripItem(trip);
                                },
                              ),
                            ],
                          );
                        },
                      ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatDateHeader(String dateKey) {
    final date = DateTime.parse(dateKey);
    final now = DateTime.now();

    if (dateKey == DateFormat('yyyy-MM-dd').format(now)) {
      return 'Today';
    } else if (dateKey ==
        DateFormat(
          'yyyy-MM-dd',
        ).format(now.subtract(const Duration(days: 1)))) {
      return 'Yesterday';
    } else {
      return DateFormat('EEEE, MMMM d').format(date);
    }
  }

  Widget _buildTripItem(TripData trip) {
    final textTheme = CupertinoTheme.of(context).textTheme;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
      decoration: BoxDecoration(
        color: CupertinoColors.systemBackground,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: CupertinoColors.systemGrey4),
        boxShadow: [
          BoxShadow(
            color: CupertinoColors.systemGrey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Time and activity
          Row(
            children: [
              Text(
                '${DateFormat('h:mm a').format(trip.startTime)} - ${DateFormat('h:mm a').format(trip.endTime)}',
                style: textTheme.textStyle.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Spacer(),
              _buildActivityIcon(trip.activityType),
              const SizedBox(width: 4.0),
              Text(
                _formatActivityType(trip.activityType),
                style: textTheme.tabLabelTextStyle.copyWith(
                  color: CupertinoColors.systemGrey,
                ),
              ),
            ],
          ),

          const SizedBox(height: 12.0),

          // Places
          Row(
            children: [
              const Icon(
                CupertinoIcons.location_solid,
                size: 16.0,
                color: CupertinoColors.activeBlue,
              ),
              const SizedBox(width: 8.0),
              Expanded(
                child: Text(
                  '${trip.startPlace} → ${trip.endPlace}',
                  style: textTheme.textStyle,
                ),
              ),
            ],
          ),

          const SizedBox(height: 12.0),

          // Stats
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStat(
                icon: CupertinoIcons.arrow_right_arrow_left,
                label: '${trip.distance.toStringAsFixed(1)} km',
              ),
              _buildStat(
                icon: CupertinoIcons.time,
                label: '${trip.duration} min',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActivityIcon(String activityType) {
    IconData icon;
    Color color;

    switch (activityType) {
      case 'walking':
        icon = CupertinoIcons.nosign;
        color = CupertinoColors.activeGreen;
        break;
      case 'running':
        icon = CupertinoIcons.nosign;
        color = CupertinoColors.systemOrange;
        break;
      case 'in_vehicle':
        icon = CupertinoIcons.car_fill;
        color = CupertinoColors.activeBlue;
        break;
      case 'on_bicycle':
        icon = CupertinoIcons.nosign;
        color = CupertinoColors.systemRed;
        break;
      case 'still':
        icon = CupertinoIcons.person_solid;
        color = CupertinoColors.systemGrey;
        break;
      default:
        icon = CupertinoIcons.question_circle;
        color = CupertinoColors.systemGrey;
        break;
    }

    return Icon(icon, size: 16.0, color: color);
  }

  String _formatActivityType(String activityType) {
    switch (activityType) {
      case 'walking':
        return 'Walking';
      case 'running':
        return 'Running';
      case 'in_vehicle':
        return 'Driving';
      case 'on_bicycle':
        return 'Cycling';
      case 'still':
        return 'Stationary';
      default:
        return 'Unknown';
    }
  }

  Widget _buildStat({required IconData icon, required String label}) {
    final textTheme = CupertinoTheme.of(context).textTheme;

    return Row(
      children: [
        Icon(icon, size: 16.0, color: CupertinoColors.systemGrey),
        const SizedBox(width: 4.0),
        Text(
          label,
          style: textTheme.tabLabelTextStyle.copyWith(
            color: CupertinoColors.systemGrey,
          ),
        ),
      ],
    );
  }
}

class TripData {
  final String id;
  final DateTime startTime;
  final DateTime endTime;
  final String startPlace;
  final String endPlace;
  final String activityType;
  final double distance; // in kilometers
  final int duration; // in minutes

  TripData({
    required this.id,
    required this.startTime,
    required this.endTime,
    required this.startPlace,
    required this.endPlace,
    required this.activityType,
    required this.distance,
    required this.duration,
  });
}
