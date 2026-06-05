import time

STEP = 3


def smooth_move(current, target):
    if current < target - STEP:
        return current + STEP
    if current > target + STEP:
        return current - STEP
    return target


def set_servo_angle(servo_pwm, channel, angle_current, angle_target):
    angle_current = int(angle_current)
    angle_target = int(angle_target)

    while True:
        angle_current = smooth_move(angle_current, angle_target)
        servo_pwm.set_pwm(channel, 0, angle_current)
        if angle_current == angle_target:
            break
    return angle_current


def servo_control(servo_pwm, channel, angle_current, angle_max, angle_min, pixel_delta):
    """Move one servo axis according to the target-center pixel offset."""
    angle_target = angle_current - pixel_delta * 0.08
    angle_target = max(angle_min, min(angle_max, angle_target))
    return set_servo_angle(servo_pwm, channel, angle_current, angle_target)
