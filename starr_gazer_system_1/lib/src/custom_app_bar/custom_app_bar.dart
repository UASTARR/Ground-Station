import 'package:flutter/material.dart';
import 'package:starr_gazer_system_1/src/custom_app_bar/clock_widget.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  const CustomAppBar({
    super.key,
    required this.title,
    required this.actions,
  });
  final String title;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    return AppBar(
      leading: const Center(
        child: ClockWidget()
      ),
      leadingWidth: 250,
      title: Text(title),
      centerTitle: true,
      actions: actions,
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
