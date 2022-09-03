#ifndef __ROTATOR_H
#define __ROTATOR_H

namespace Sun {
    class Rotator {
    public:
        /** Rotation around the right axis (around Y axis), Looking up and down (0=Straight Ahead, +Up, -Down) */
        float _pitch;

        /** Rotation around the up axis (around Z axis), Running in circles 0=East, +North, -South. */
        float _yaw;

        /** Rotation around the forward axis (around X axis), Tilting your head, 0=Straight, +Clockwise, -CCW. */
        float _roll;
    };
    
}
#endif
