#include "../Engine.h"

EyeSpace ToEyeSpace(Vector3 worldSpace, Transform cameraTransform) {
    // TODO: implement this. Currently not implemented as I have no API for
    // quaternion rotations and translations about a non-origin point
    return worldSpace;
}

HomoClipSpace ToClipSpace(EyeSpace eyeSpace, FrameConstants constants) {
    const float f = constants.distanceScaling;
    const float af = constants.effectiveAspectRatio * f;
    const float q = constants.frustum;
    const float negZNearQ = -constants.zNear * q;

    // clang-format off
    Matrix4x4 projectionMatrix = {
        {af, 0,         0, 0},
        { 0, f,         0, 0},
        { 0, 0,         q, 1},
        { 0, 0, negZNearQ, 0},
    }; // clang-format on

    Vector4 projectionCoordinate = {eyeSpace.x, eyeSpace.y, eyeSpace.z, 1};
    return Vector4MulMatrix4x4(projectionCoordinate, projectionMatrix);
}

DeviceSpace ToNormalDeviceSpace(HomoClipSpace clipSpace) {
    float x = clipSpace.x;
    float y = clipSpace.y;
    float z = clipSpace.z;

    Vector3 coordinate = (Vector3) {x, y, z};

    DeviceSpace result;

    if (!Approx(clipSpace.w, 0)) {
        float inverseZ = 1.0f / clipSpace.w;
        result = MulVector3(coordinate, inverseZ);
    } else {
        // can't divide by 0... so don't
        result = coordinate;
    }

    return result; // output: [x, y, z] where x, y, z are in [-1, 1].
}

ViewportSpace ToViewportSpace(DeviceSpace deviceSpace, FrameConstants constants) {
    return (ViewportSpace) {
        .x = (int)(0.5f * (deviceSpace.x + 1) * constants.viewportWidth),
        .y = (int)(0.5f * (deviceSpace.y + 1) * constants.viewportHeight),
        .depth = 0.5f * (deviceSpace.z + 1),
    };
}
