// OpenXR backend - ERVR's own XrSession, bound to the game's D3D12 device and queue.
//
// This replaces the old parasitic LibOVR/OpenVR backends that captured R.E.A.L.'s session.
// Two planes live here:
//   * pose/input  - called from the worker thread at PollHz, feeding GetLatestVRState().
//   * frame loop  - called from the render thread (inside ReShade's present event) by
//                   stereo.cpp only: WaitBeginFrame / LocateViews / Acquire / EndFrame.
// OpenXR requires external synchronisation on the session, so every session-touching call
// takes g_mtx. xrWaitFrame is deliberately made outside the lock: it blocks until the
// runtime's next frame cadence and would otherwise stall the worker's input polling.
#include "vr_backend.h"
#include "config.h"
#include "log.h"

#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D12
#include <windows.h>
#include <d3d12.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace ervr {
namespace {

std::recursive_mutex g_mtx;

const char* XrStr(XrResult r) {
    switch (r) {
    case XR_SUCCESS: return "XR_SUCCESS";
    case XR_SESSION_LOSS_PENDING: return "XR_SESSION_LOSS_PENDING";
    case XR_ERROR_RUNTIME_FAILURE: return "XR_ERROR_RUNTIME_FAILURE";
    case XR_ERROR_INSTANCE_LOST: return "XR_ERROR_INSTANCE_LOST";
    case XR_ERROR_SESSION_LOST: return "XR_ERROR_SESSION_LOST";
    case XR_ERROR_FORM_FACTOR_UNAVAILABLE: return "XR_ERROR_FORM_FACTOR_UNAVAILABLE";
    case XR_ERROR_FORM_FACTOR_UNSUPPORTED: return "XR_ERROR_FORM_FACTOR_UNSUPPORTED";
    case XR_ERROR_EXTENSION_NOT_PRESENT: return "XR_ERROR_EXTENSION_NOT_PRESENT";
    case XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING: return "XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING";
    case XR_ERROR_GRAPHICS_DEVICE_INVALID: return "XR_ERROR_GRAPHICS_DEVICE_INVALID";
    case XR_ERROR_VALIDATION_FAILURE: return "XR_ERROR_VALIDATION_FAILURE";
    case XR_ERROR_CALL_ORDER_INVALID: return "XR_ERROR_CALL_ORDER_INVALID";
    case XR_ERROR_SESSION_NOT_RUNNING: return "XR_ERROR_SESSION_NOT_RUNNING";
    case XR_ERROR_SESSION_NOT_READY: return "XR_ERROR_SESSION_NOT_READY";
    case XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED: return "XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED";
    case XR_ERROR_INITIALIZATION_FAILED: return "XR_ERROR_INITIALIZATION_FAILED (runtime present but could not start - headset not streaming?)";
    case XR_ERROR_API_VERSION_UNSUPPORTED: return "XR_ERROR_API_VERSION_UNSUPPORTED";
    case XR_ERROR_OUT_OF_MEMORY: return "XR_ERROR_OUT_OF_MEMORY";
    case XR_ERROR_FUNCTION_UNSUPPORTED: return "XR_ERROR_FUNCTION_UNSUPPORTED";
    case XR_ERROR_FEATURE_UNSUPPORTED: return "XR_ERROR_FEATURE_UNSUPPORTED";
    case XR_ERROR_LIMIT_REACHED: return "XR_ERROR_LIMIT_REACHED";
    case XR_ERROR_RUNTIME_UNAVAILABLE: return "XR_ERROR_RUNTIME_UNAVAILABLE (no OpenXR runtime registered / not running)";
    case XR_ERROR_NAME_INVALID: return "XR_ERROR_NAME_INVALID";
    case XR_ERROR_HANDLE_INVALID: return "XR_ERROR_HANDLE_INVALID";
    default: return "XR_ERROR (unmapped)";
    }
}
#define XRC(expr, what)                                                                   \
    do {                                                                                  \
        XrResult _r = (expr);                                                             \
        if (XR_FAILED(_r)) { LOGE("[xr] %s failed: %s (%d)", what, XrStr(_r), (int)_r); return false; } \
    } while (0)

// Where the render thread waits inside the runtime besides xrWaitFrame (which the hb timing
// line reports): xrBeginFrame, xrEndFrame and the swapchain-image waits, summed per 5 s at
// LogLevel=debug. Render thread only. Added 2026-11-03: alternate-eye frames spent ~3.6 ms a
// frame in our present beyond xrWaitFrame, and these say which call holds it.
struct XrCallClock {
    double sumMs = 0, maxMs = 0;
    long n = 0;
    void Add(double ms) { sumMs += ms; if (ms > maxMs) maxMs = ms; ++n; }
    double Avg() const { return n ? sumMs / n : 0.0; }
};
XrCallClock g_clkBegin, g_clkEnd, g_clkImage;
double QpcMs() {
    static const double f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return (double)x.QuadPart; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / f;
}

inline Vec3  ToVec(const XrVector3f& v) { return {v.x, v.y, v.z}; }
inline Quat  ToQuat(const XrQuaternionf& q) { return {q.x, q.y, q.z, q.w}; }
inline VRFov ToFov(const XrFovf& f) { return {f.angleLeft, f.angleRight, f.angleUp, f.angleDown}; }
inline XrFovf FromFov(const VRFov& f) { return {f.angleLeft, f.angleRight, f.angleUp, f.angleDown}; }
inline XrPosef FromPose(const Pose& p) {
    XrPosef o{};
    o.orientation = {p.rot.x, p.rot.y, p.rot.z, p.rot.w};
    o.position = {p.pos.x, p.pos.y, p.pos.z};
    return o;
}

struct Swap {
    XrSwapchain handle = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageD3D12KHR> images;
    uint32_t w = 0, h = 0;
    int64_t  format = 0;
    uint32_t acquired = UINT32_MAX;

    void Destroy() {
        if (handle != XR_NULL_HANDLE) xrDestroySwapchain(handle);
        handle = XR_NULL_HANDLE; images.clear(); acquired = UINT32_MAX; w = h = 0;
    }
};

// A quad's lazy world anchor. World-locked content is what the compositor can reproject
// perfectly at any submit rate; view-locked content has to move WITH the head between
// submits, which reads as judder the moment submissions run below the display rate (they
// always do here, at one XR frame per stereo pair). So the quads are placed in the world
// and simply re-placed whenever the head has turned or walked far enough to lose them.
struct Anchor {
    bool  valid = false;
    Vec3  origin;      // where the head was when the quad was last placed
    Quat  yaw;         // the head's yaw at that moment, flattened to the horizon
    Vec3  fwd;         // and the forward it implies, kept for the re-anchor test
    float dist = 0.f;
};

// ---- action set ---------------------------------------------------------------
struct Actions {
    XrActionSet set = XR_NULL_HANDLE;
    XrAction pose[2]{}, trigger[2]{}, squeeze[2]{}, stick[2]{}, stickClick[2]{}, haptic[2]{};
    // Valve Index only: the trackpads (where the thumb is, how hard it presses) and the grip's
    // squeeze force. No other profile binds them, so they read zero / inactive elsewhere.
    XrAction trackpad[2]{}, trackpadForce[2]{}, squeezeForce[2]{};
    XrAction btnA = XR_NULL_HANDLE, btnB = XR_NULL_HANDLE, btnX = XR_NULL_HANDLE,
             btnY = XR_NULL_HANDLE, btnMenu = XR_NULL_HANDLE;
    XrSpace  handSpace[2]{};
};

class OpenXRBackend final : public IVRBackend {
public:
    const char* Name() const override { return "openxr"; }

    // ---- attach: instance + system only (no graphics yet) ----
    bool TryAttach() override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (instance_ != XR_NULL_HANDLE && systemId_ != XR_NULL_SYSTEM_ID) return true;
        // The worker calls this at PollHz (250 Hz). Creating an instance is expensive and
        // talks to the runtime over IPC, so only retry every couple of seconds.
        const uint64_t now = GetTickCount64();
        if (now - lastAttach_ < 2000) return false;
        lastAttach_ = now;
        if (!CreateInstance()) return false;
        if (!GetSystem()) return false;
        return true;
    }
    bool IsReady() const override { return instance_ != XR_NULL_HANDLE && systemId_ != XR_NULL_SYSTEM_ID; }

    void Detach() override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        TeardownSession();
        if (instance_ != XR_NULL_HANDLE) { xrDestroyInstance(instance_); instance_ = XR_NULL_HANDLE; }
        systemId_ = XR_NULL_SYSTEM_ID;
    }

    // ---- graphics binding: creates the session, spaces, actions, eye swapchains ----
    bool BindGraphics(ID3D12Device* device, ID3D12CommandQueue* queue) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (bound_) return true;
        if (!IsReady() || !device || !queue) return false;

        // The runtime must be asked for its requirements before session creation, even
        // though we are handing it a device the game already made.
        auto pfnReq = GetProc<PFN_xrGetD3D12GraphicsRequirementsKHR>("xrGetD3D12GraphicsRequirementsKHR");
        if (!pfnReq) { LOGE("[xr] xrGetD3D12GraphicsRequirementsKHR missing (is XR_KHR_D3D12_enable supported?)"); return false; }
        XrGraphicsRequirementsD3D12KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_D3D12_KHR};
        XRC(pfnReq(instance_, systemId_, &req), "xrGetD3D12GraphicsRequirementsKHR");

        LUID want = req.adapterLuid, have = device->GetAdapterLuid();
        if (want.LowPart != have.LowPart || want.HighPart != have.HighPart)
            LOGW("[xr] runtime prefers adapter LUID %08lx:%08lx but the game renders on %08lx:%08lx; "
                 "submission may be cross-adapter (slow)", want.HighPart, want.LowPart, have.HighPart, have.LowPart);

        XrGraphicsBindingD3D12KHR bind{XR_TYPE_GRAPHICS_BINDING_D3D12_KHR};
        bind.device = device;
        bind.queue = queue;
        XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
        sci.next = &bind;
        sci.systemId = systemId_;
        XRC(xrCreateSession(instance_, &sci, &session_), "xrCreateSession");
        LOGI("[xr] session created on device %p queue %p", (void*)device, (void*)queue);

        if (!CreateSpaces()) return false;
        if (!CreateActions()) return false;
        device_ = device; queue_ = queue;
        bound_ = true;
        return true;
    }
    bool GraphicsBound() const override { return bound_; }

    void UnbindGraphics(ID3D12Device* device) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || device != device_) return;
        LOGI("[xr] the game is destroying the D3D12 device the session is bound to - ending the session first");
        TeardownSession();
    }

    bool GetRecommendedEyeSize(uint32_t& w, uint32_t& h) const override {
        if (viewConfig_.empty()) return false;
        w = viewConfig_[0].recommendedImageRectWidth;
        h = viewConfig_[0].recommendedImageRectHeight;
        return w > 0 && h > 0;
    }
    DXGI_FORMAT EyeFormat() const override { return (DXGI_FORMAT)eyeFormat_; }
    DXGI_FORMAT QuadFormat() const override { return (DXGI_FORMAT)quad_.format; }
    DXGI_FORMAT HudFormat() const override { return (DXGI_FORMAT)hud_.format; }
    DXGI_FORMAT OverlayFormat(int which) const override {
        return which >= 0 && which < OverlayCount ? (DXGI_FORMAT)overlays_[which].format : DXGI_FORMAT_UNKNOWN;
    }

    bool EnsureEyeSwapchains(uint32_t w, uint32_t h) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || w == 0 || h == 0) return false;
        if (!viewConfig_.empty()) {
            w = std::min(w, viewConfig_[0].maxImageRectWidth);
            h = std::min(h, viewConfig_[0].maxImageRectHeight);
        }
        if (eyes_[0].handle != XR_NULL_HANDLE) {
            // Only rebuild on a real change: the crop moves by a pixel or two as the runtime
            // re-predicts the eye poses, and tearing down swapchains every frame would stall.
            const uint32_t dw = eyes_[0].w > w ? eyes_[0].w - w : w - eyes_[0].w;
            const uint32_t dh = eyes_[0].h > h ? eyes_[0].h - h : h - eyes_[0].h;
            if (dw * 32 < eyes_[0].w && dh * 32 < eyes_[0].h) return true;
        }
        for (auto& e : eyes_) e.Destroy();
        if (eyeFormat_ == 0)
            eyeFormat_ = PickFormat({DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                                     DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM});
        if (eyeFormat_ == 0) { LOGE("[xr] no usable eye swapchain format"); return false; }
        for (int i = 0; i < 2; ++i)
            if (!CreateSwap(eyes_[i], w, h, eyeFormat_)) return false;
        LOGI("[xr] eye swapchains %ux%u fmt %lld (%u images)", w, h, (long long)eyeFormat_,
             (unsigned)eyes_[0].images.size());
        return true;
    }

    bool EnsureQuadSwapchain(uint32_t w, uint32_t h, DXGI_FORMAT bbFormat) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || w == 0 || h == 0) return false;
        if (quad_.handle != XR_NULL_HANDLE && quad_.w == w && quad_.h == h) return true;
        quad_.Destroy();
        // Deliberately NOT "whatever is copy-compatible with the back buffer". A raw copy
        // preserves the bits, and the game's bits are gamma encoded while OpenXR reads a
        // non-sRGB swapchain as linear light - that mismatch is what washed the picture out.
        // The blitter converts in its shader instead, so the only thing that matters here is
        // precision: R16F first, so ELDEN RING's 10-bit output survives without banding.
        int64_t fmt = PickFormat({DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                  DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM});
        if (fmt == 0) { LOGE("[xr] runtime offers no usable quad format"); return false; }
        if (!CreateSwap(quad_, w, h, fmt)) return false;
        LOGI("[xr] quad swapchain %ux%u fmt %lld (back buffer fmt %u)", w, h, (long long)fmt, (unsigned)bbFormat);
        return true;
    }

    bool EnsureHudSwapchain(uint32_t w, uint32_t h) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || w == 0 || h == 0) return false;
        if (hud_.handle != XR_NULL_HANDLE && hud_.w == w && hud_.h == h) return true;
        hud_.Destroy();
        // Unlike every other swapchain here, this one's ALPHA is the payload: the layer is
        // blended over the world, so a format whose alpha the runtime would ignore (or
        // quantise to two bits, as the game's own R10G10B10A2 back buffer does) defeats the
        // whole point. Eight bits of alpha minimum, in that order of preference.
        int64_t fmt = PickFormat({DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                                  DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM,
                                  DXGI_FORMAT_B8G8R8A8_UNORM});
        if (fmt == 0) { LOGE("[xr] runtime offers no usable HUD format"); return false; }
        if (!CreateSwap(hud_, w, h, fmt)) return false;
        LOGI("[xr] HUD swapchain %ux%u fmt %lld", w, h, (long long)fmt);
        return true;
    }

    bool EnsureOverlaySwapchain(int which, uint32_t w, uint32_t h) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || which < 0 || which >= OverlayCount || w == 0 || h == 0) return false;
        Swap& s = overlays_[which];
        if (s.handle != XR_NULL_HANDLE && s.w == w && s.h == h) return true;
        s.Destroy();
        // Alpha-blended like the HUD. The image is composed on the CPU in 8-bit sRGB, so those
        // formats take a straight copy; the others are converted on upload.
        int64_t fmt = PickFormat({DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                                  DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
                                  DXGI_FORMAT_B8G8R8A8_UNORM});
        if (fmt == 0) { LOGE("[xr] runtime offers no usable overlay format"); return false; }
        if (!CreateSwap(s, w, h, fmt)) return false;
        LOGI("[xr] overlay %d swapchain %ux%u fmt %lld", which, w, h, (long long)fmt);
        return true;
    }

    // ---- frame loop (render thread) ----
    bool WaitBeginFrame(VRFrameInfo& out) override {
        if (!bound_) return false;
        PumpEvents();
        if (!sessionRunning_) return false;

        // Runtime-death failsafe: a broken runtime (VDXR losing the display to an exclusive
        // fullscreen switch, 2026-08-21, or the streamer dropping) fails EVERY frame; the
        // old behaviour spammed it until something crashed the game. After ~2 s of
        // consecutive failures, give up on stereo for this session and let the game run
        // flat - the user fixes the display state and relaunches.
        static std::atomic<int>  s_frameFails{0};
        static std::atomic<bool> s_runtimeDead{false};
        if (s_runtimeDead.load(std::memory_order_relaxed)) return false;
        auto frameFail = [&](const char* what, XrResult rr) {
            LOGW("[xr] %s: %s", what, XrStr(rr));
            if (s_frameFails.fetch_add(1, std::memory_order_relaxed) + 1 >= 120 &&
                !s_runtimeDead.exchange(true))
                LOGE("[xr] runtime failed 120 consecutive frames - STEREO DISABLED for this "
                     "session, game continues flat. Fix the runtime/display (no exclusive "
                     "fullscreen with Virtual Desktop!) and restart for VR.");
            return false;
        };

        XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState fs{XR_TYPE_FRAME_STATE};
        LARGE_INTEGER w0, w1, qf;
        QueryPerformanceCounter(&w0);
        XrResult r = xrWaitFrame(session_, &wi, &fs);   // deliberately outside the lock
        QueryPerformanceCounter(&w1);
        if (XR_FAILED(r)) return frameFail("xrWaitFrame", r);
        QueryPerformanceFrequency(&qf);
        out.waitMs = (float)((double)(w1.QuadPart - w0.QuadPart) * 1000.0 / (double)qf.QuadPart);
        out.displayPeriodNs = fs.predictedDisplayPeriod;

        std::lock_guard<std::recursive_mutex> g(g_mtx);
        XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
        const double b0 = QpcMs();
        r = xrBeginFrame(session_, &bi);
        g_clkBegin.Add(QpcMs() - b0);
        if (XR_FAILED(r) && r != XR_FRAME_DISCARDED) return frameFail("xrBeginFrame", r);
        s_frameFails.store(0, std::memory_order_relaxed);

        displayTime_.store(fs.predictedDisplayTime);
        frameBegun_ = true;
        out.displayTimeNs = fs.predictedDisplayTime;
        out.displayTimeSec = (double)fs.predictedDisplayTime * 1e-9;
        out.shouldRender = fs.shouldRender != XR_FALSE;
        out.pairIndex = ++pairIndex_;
        return true;
    }

    bool LocateViews(int64_t displayTimeNs, VREyeView out[2]) override {
        return LocateIn(appSpace_, displayTimeNs, out);
    }
    // Locating the views in VIEW space gives the eyes relative to the head directly - no
    // second xrLocateSpace and no subtraction of two poses sampled at different instants.
    bool LocateEyeOffsets(int64_t displayTimeNs, VREyeView out[2]) override {
        return LocateIn(viewSpace_, displayTimeNs, out);
    }

    ID3D12Resource* AcquireEyeImage(VREye eye) override { return AcquireFrom(eyes_[eye]); }
    void ReleaseEyeImage(VREye eye) override { ReleaseTo(eyes_[eye]); }
    ID3D12Resource* AcquireQuadImage() override { return AcquireFrom(quad_); }
    void ReleaseQuadImage() override { ReleaseTo(quad_); }
    ID3D12Resource* AcquireHudImage() override { return AcquireFrom(hud_); }
    void ReleaseHudImage() override { ReleaseTo(hud_); }
    ID3D12Resource* AcquireOverlayImage(int which) override {
        return which >= 0 && which < OverlayCount ? AcquireFrom(overlays_[which]) : nullptr;
    }
    void ReleaseOverlayImage(int which) override { if (which >= 0 && which < OverlayCount) ReleaseTo(overlays_[which]); }

    // Places a quad layer in space.
    //
    // Body/World locks put the quad in the app (stage) space and re-place it only when the
    // head has moved away from it - the compositor can then reproject it perfectly however
    // late the submission is, which head-locking cannot survive at one XR frame per stereo
    // pair. Head lock is kept for the case where the player wants the HUD welded to their
    // face, and as the fallback whenever tracking is not reportable.
    void PlaceQuad(XrCompositionLayerQuad& q, Anchor& a, int lock, float distM,
                   float followDeg, float followRadiusM, float heightOffsetM,
                   float yawDeg, float pitchDeg, bool recenter) {
        const Quat off = (Quat::YawQ(Deg2Rad(yawDeg)) * Quat::AxisAngle({1, 0, 0}, Deg2Rad(pitchDeg)))
                             .Normalized();
        // The quad hangs `distM` along the anchor's forward, offset by the player's own
        // yaw/pitch/height nudges. Rotating first and translating after is what makes the
        // yaw offset swing the panel around the player rather than slide it sideways.
        auto place = [&](const Vec3& origin, const Quat& yawq) {
            const Quat rot = (yawq * off).Normalized();
            const Vec3 pos = origin + rot.Rotate({0.f, 0.f, -distM}) + Vec3{0.f, heightOffsetM, 0.f};
            q.pose.orientation = {rot.x, rot.y, rot.z, rot.w};
            q.pose.position = {pos.x, pos.y, pos.z};
        };
        auto headLock = [&] {
            a.valid = false;
            q.space = viewSpace_;
            place({}, Quat{});
        };
        if (lock == QuadLockHead) { headLock(); return; }

        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
        const bool located =
            XR_SUCCEEDED(xrLocateSpace(viewSpace_, appSpace_, displayTime_.load(), &loc)) &&
            (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) &&
            (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT);
        Vec3 fwd;
        if (located) {
            const XrQuaternionf& hq = loc.pose.orientation;
            Vec3 f = Quat{hq.x, hq.y, hq.z, hq.w}.Forward();
            f.y = 0.f;
            if (f.Length() > 0.2f) fwd = f.Normalized();   // else: staring straight up or down
        }
        if (fwd.Length() < 0.5f) {
            // No usable heading. Holding the existing anchor is right if there is one -
            // the panel simply stays where it was - and only a cold start falls back to
            // the head.
            if (!a.valid) { headLock(); return; }
            q.space = appSpace_;
            place(a.origin, a.yaw);
            return;
        }

        const Vec3 hp{loc.pose.position.x, loc.pose.position.y, loc.pose.position.z};
        bool replace = !a.valid || recenter || a.dist != distM;
        if (!replace && lock != QuadLockWorld) {
            const float dx = hp.x - a.origin.x, dz = hp.z - a.origin.z;
            replace = fwd.x * a.fwd.x + fwd.z * a.fwd.z < std::cos(Deg2Rad(followDeg)) ||
                      dx * dx + dz * dz > followRadiusM * followRadiusM ||
                      std::fabs(hp.y - a.origin.y) > 0.5f;
        }
        if (replace) {
            a.valid = true;
            a.origin = hp;
            a.fwd = fwd;
            a.dist = distM;
            a.yaw = Quat::YawQ(std::atan2(-fwd.x, -fwd.z));
        }
        q.space = appSpace_;
        place(a.origin, a.yaw);
    }

    bool EndFrame(const VRSubmitLayer& layer) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || !frameBegun_) return false;
        frameBegun_ = false;

        XrCompositionLayerProjectionView pv[2]{};
        XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        XrCompositionLayerQuad hudQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        XrCompositionLayerQuad overlayQuad[OverlayCount]{};
        const XrCompositionLayerBaseHeader* layers[3 + OverlayCount]{};
        uint32_t count = 0;

        if (layer.projectionValid && eyes_[0].handle != XR_NULL_HANDLE) {
            for (int i = 0; i < 2; ++i) {
                pv[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
                pv[i].pose = FromPose(layer.views[i].pose);
                pv[i].fov = FromFov(layer.views[i].fov);
                pv[i].subImage.swapchain = eyes_[i].handle;
                pv[i].subImage.imageArrayIndex = 0;
                pv[i].subImage.imageRect = {{0, 0}, {(int32_t)eyes_[i].w, (int32_t)eyes_[i].h}};
            }
            proj.space = appSpace_;
            proj.layerFlags = 0;   // the world is opaque; the game's alpha channel is not coverage
            proj.viewCount = 2;
            proj.views = pv;
            layers[count++] = (const XrCompositionLayerBaseHeader*)&proj;
        }
        if (layer.quadValid && quad_.handle != XR_NULL_HANDLE) {
            const float aspect = quad_.h ? (float)quad_.w / (float)quad_.h : 1.777f;
            quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            quad.subImage.swapchain = quad_.handle;
            quad.subImage.imageArrayIndex = 0;
            quad.subImage.imageRect = {{0, 0}, {(int32_t)quad_.w, (int32_t)quad_.h}};
            quad.size = {layer.quadWidthM, layer.quadWidthM / aspect};
            quad.layerFlags = 0;   // the cinema screen is the whole picture: opaque
            // ~41 deg of head turn or half a metre of walking re-places the screen.
            PlaceQuad(quad, cinemaAnchor_, QuadLockBody, layer.quadDistanceM,
                      41.f, 0.5f, 0.f, 0.f, 0.f, false);
            layers[count++] = (const XrCompositionLayerBaseHeader*)&quad;
        }
        if (layer.hudValid && hud_.handle != XR_NULL_HANDLE) {
            const float aspect = hud_.h ? (float)hud_.w / (float)hud_.h : 1.f;
            hudQuad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            hudQuad.subImage.swapchain = hud_.handle;
            hudQuad.subImage.imageArrayIndex = 0;
            hudQuad.subImage.imageRect = {{0, 0}, {(int32_t)hud_.w, (int32_t)hud_.h}};
            hudQuad.size = {layer.hudWidthM, layer.hudWidthM / aspect};
            // The one layer here whose alpha means something. The UI was drawn over a
            // transparent clear with ordinary src-over blending, which leaves the colour
            // already multiplied by coverage - so PREMULTIPLIED (i.e. no UNPREMULTIPLIED
            // flag) is the correct reading, and it is also the one that does not fringe
            // dark halos around white text.
            hudQuad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            PlaceQuad(hudQuad, hudAnchor_, layer.hudLock, layer.hudDistanceM,
                      layer.hudFollowDeg, layer.hudFollowRadiusM, layer.hudHeightOffsetM,
                      layer.hudYawDeg, layer.hudPitchDeg, layer.hudRecenter);
            layers[count++] = (const XrCompositionLayerBaseHeader*)&hudQuad;
        }
        // The overlays carry their own stage-space pose: they belong to the hands, not the view.
        for (int i = 0; i < OverlayCount; ++i) {
            const VROverlayQuad& o = layer.overlays[i];
            if (!o.valid || overlays_[i].handle == XR_NULL_HANDLE || o.widthM <= 0.f || o.heightM <= 0.f) continue;
            if (maxLayers_ && count >= maxLayers_) break;   // over the runtime's limit xrEndFrame would refuse the whole frame
            XrCompositionLayerQuad& q = overlayQuad[i];
            q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
            q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;   // premultiplied, as the HUD
            q.space = appSpace_;
            q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            q.subImage.swapchain = overlays_[i].handle;
            q.subImage.imageArrayIndex = 0;
            q.subImage.imageRect = {{0, 0}, {(int32_t)overlays_[i].w, (int32_t)overlays_[i].h}};
            q.pose = FromPose(o.pose);
            q.size = {o.widthM, o.heightM};
            layers[count++] = (const XrCompositionLayerBaseHeader*)&q;
        }

        XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
        ei.displayTime = displayTime_.load();
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        ei.layerCount = count;
        ei.layers = count ? layers : nullptr;
        const double e0 = QpcMs();
        XrResult r = xrEndFrame(session_, &ei);
        const double e1 = QpcMs();
        g_clkEnd.Add(e1 - e0);
        static double s_clkLog = 0;
        if (e1 - s_clkLog > 5000.0) {
            if (s_clkLog > 0)
                LOGD("[xr] 5s runtime waits ms avg/max: xrBeginFrame %.2f/%.2f, xrEndFrame %.2f/%.2f, "
                     "swapchain image %.2f/%.2f (%ld frames, %ld images)",
                     g_clkBegin.Avg(), g_clkBegin.maxMs, g_clkEnd.Avg(), g_clkEnd.maxMs,
                     g_clkImage.Avg(), g_clkImage.maxMs, g_clkEnd.n, g_clkImage.n);
            s_clkLog = e1;
            g_clkBegin = g_clkEnd = g_clkImage = XrCallClock{};
        }
        if (XR_FAILED(r)) {
            static XrResult s_last = XR_SUCCESS;
            if (s_last != r) {
                s_last = r;
                char resultName[XR_MAX_RESULT_STRING_SIZE]{};
                const XrResult named = xrResultToString(instance_, r, resultName);
                LOGW("[xr] xrEndFrame: %s (%d), runtime=%s | layers=%u projection=%d cinema=%d hud=%d displayTime=%lld",
                     XrStr(r), (int)r,
                     XR_SUCCEEDED(named) ? resultName : "(unavailable)",
                     count, (int)layer.projectionValid, (int)layer.quadValid,
                     (int)layer.hudValid, (long long)ei.displayTime);
            }
            return false;
        }
        return true;
    }

    // ---- pose/input plane (worker thread) ----
    bool Poll(VRTracking& tr, VRInputState& in) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (session_ == XR_NULL_HANDLE || !actions_.set) return false;
        if (sessionState_ != XR_SESSION_STATE_FOCUSED && sessionState_ != XR_SESSION_STATE_VISIBLE) return false;

        XrActiveActionSet active{actions_.set, XR_NULL_PATH};
        XrActionsSyncInfo si{XR_TYPE_ACTIONS_SYNC_INFO};
        si.countActiveActionSets = 1;
        si.activeActionSets = &active;
        XrResult r = xrSyncActions(session_, &si);
        if (XR_FAILED(r)) return false;
        if (r == XR_SESSION_NOT_FOCUSED) return false;

        const XrTime t = SampleTime();
        tr.time = (double)t * 1e-9;
        tr.head = LocateSpace(viewSpace_, t);
        for (int h = 0; h < 2; ++h) {
            tr.hand[h] = LocateSpace(actions_.handSpace[h], t);
            in.indexTrigger[h] = GetFloat(actions_.trigger[h]);
            in.handTrigger[h]  = GetFloat(actions_.squeeze[h]);
            // Valve Index: squeeze/value is the grip's touch sensor, which reads closed as soon as the
            // fingers rest on the handle - "only touching the grip brings up the weapon swap" (2026-11-02
            // report). The squeeze FORCE is what a deliberate grip is; IndexGripForce reads as a full one.
            bool forceBound = false;
            const float force = GetFloat(actions_.squeezeForce[h], &forceBound);
            if (forceBound && Cfg().indexGripForce > 0.f) in.handTrigger[h] = std::min(1.f, force / Cfg().indexGripForce);
            const XrVector2f pad = GetVec2(actions_.trackpad[h]);
            in.pad[h][0] = pad.x;
            in.pad[h][1] = pad.y;
            in.padForce[h] = GetFloat(actions_.trackpadForce[h]);
            XrVector2f v = GetVec2(actions_.stick[h]);
            in.thumb[h][0] = v.x;
            in.thumb[h][1] = v.y;
            in.controllerConnected[h] = tr.hand[h].valid || in.indexTrigger[h] > 0.01f ||
                                        in.handTrigger[h] > 0.01f || std::fabs(v.x) > 0.01f || std::fabs(v.y) > 0.01f;
        }
        uint32_t b = 0;
        if (GetBool(actions_.btnA)) b |= VRB_A;
        if (GetBool(actions_.btnB)) b |= VRB_B;
        if (GetBool(actions_.btnX)) b |= VRB_X;
        if (GetBool(actions_.btnY)) b |= VRB_Y;
        if (GetBool(actions_.btnMenu)) b |= VRB_Menu;
        if (GetBool(actions_.stickClick[HandLeft]))  b |= VRB_LThumb;
        if (GetBool(actions_.stickClick[HandRight])) b |= VRB_RThumb;
        in.buttons = b;

        LogInputCensus(tr.time);

        // Event-driven: a 2 s poll misses every stick push a player actually makes, which is
        // exactly what made the 2026-08-21 "menu will not move" report unreadable (25 samples,
        // not one of them during a press). Anything non-idle logs within 250 ms.
        static double s_lastDump = 0;
        const bool anyInput =
            b != 0 || in.indexTrigger[0] > 0.1f || in.indexTrigger[1] > 0.1f ||
            in.handTrigger[0] > 0.1f || in.handTrigger[1] > 0.1f ||
            std::fabs(in.thumb[0][0]) > 0.1f || std::fabs(in.thumb[0][1]) > 0.1f ||
            std::fabs(in.thumb[1][0]) > 0.1f || std::fabs(in.thumb[1][1]) > 0.1f;
        const bool anyPad = in.padForce[0] > 0.1f || in.padForce[1] > 0.1f;
        if (tr.time - s_lastDump > (anyInput || anyPad ? 0.25 : 2.0)) {
            s_lastDump = tr.time;
            LOGD("[xr] raw: state %d conn(%d,%d) btn 0x%04x trig(%.2f,%.2f) grip(%.2f,%.2f) "
                 "stick L(%.2f,%.2f) R(%.2f,%.2f) pad L(%.2f,%.2f %.2f) R(%.2f,%.2f %.2f) head(%.2f,%.2f,%.2f)",
                 (int)sessionState_, in.controllerConnected[0], in.controllerConnected[1], b,
                 in.indexTrigger[0], in.indexTrigger[1], in.handTrigger[0], in.handTrigger[1],
                 in.thumb[0][0], in.thumb[0][1], in.thumb[1][0], in.thumb[1][1],
                 in.pad[0][0], in.pad[0][1], in.padForce[0], in.pad[1][0], in.pad[1][1], in.padForce[1],
                 tr.head.pos.x, tr.head.pos.y, tr.head.pos.z);
        }
        return true;
    }

    void Haptic(Hand hand, float amplitude, float durationSec) override {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (session_ == XR_NULL_HANDLE || !actions_.haptic[hand]) return;
        XrHapticActionInfo hi{XR_TYPE_HAPTIC_ACTION_INFO};
        hi.action = actions_.haptic[hand];
        if (amplitude <= 0.f) { xrStopHapticFeedback(session_, &hi); return; }
        XrHapticVibration v{XR_TYPE_HAPTIC_VIBRATION};
        v.amplitude = Clamp(amplitude, 0.f, 1.f);
        v.duration = durationSec > 0 ? (XrDuration)(durationSec * 1e9) : XR_MIN_HAPTIC_DURATION;
        v.frequency = XR_FREQUENCY_UNSPECIFIED;
        xrApplyHapticFeedback(session_, &hi, (const XrHapticBaseHeader*)&v);
    }

private:
    std::string PathStr(XrPath p) const {
        if (p == XR_NULL_PATH) return "<none>";
        char buf[XR_MAX_PATH_LENGTH]{};
        uint32_t n = 0;
        if (XR_FAILED(xrPathToString(instance_, p, sizeof(buf), &n, buf))) return "?";
        return buf;
    }
    std::string ProfileFor(const char* handPath) {
        XrInteractionProfileState ips{XR_TYPE_INTERACTION_PROFILE_STATE};
        XrPath hp = XR_NULL_PATH;
        xrStringToPath(instance_, handPath, &hp);
        if (hp == XR_NULL_PATH || XR_FAILED(xrGetCurrentInteractionProfile(session_, hp, &ips)))
            return "<none>";
        return PathStr(ips.interactionProfile);
    }

    // isActive is the runtime telling us whether an action has a bound source AT ALL, as
    // opposed to "bound but the user is not touching it". Without it, a control that never
    // binds is indistinguishable from one nobody pressed - the ambiguity that stalled the
    // 2026-08-21 "left stick does nothing, A works" report. Logged only when it changes.
    void LogInputCensus(double now) {
        if (now - lastCensus_ < 1.0) return;
        lastCensus_ = now;
        auto boolAct = [&](XrAction a) {
            if (!a) return false;
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO}; gi.action = a;
            XrActionStateBoolean s{XR_TYPE_ACTION_STATE_BOOLEAN};
            return XR_SUCCEEDED(xrGetActionStateBoolean(session_, &gi, &s)) && s.isActive != XR_FALSE;
        };
        auto floatAct = [&](XrAction a) {
            if (!a) return false;
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO}; gi.action = a;
            XrActionStateFloat s{XR_TYPE_ACTION_STATE_FLOAT};
            return XR_SUCCEEDED(xrGetActionStateFloat(session_, &gi, &s)) && s.isActive != XR_FALSE;
        };
        auto vecAct = [&](XrAction a) {
            if (!a) return false;
            XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO}; gi.action = a;
            XrActionStateVector2f s{XR_TYPE_ACTION_STATE_VECTOR2F};
            return XR_SUCCEEDED(xrGetActionStateVector2f(session_, &gi, &s)) && s.isActive != XR_FALSE;
        };
        const bool ls = vecAct(actions_.stick[0]),  rs = vecAct(actions_.stick[1]);
        const bool lt = floatAct(actions_.trigger[0]), rt = floatAct(actions_.trigger[1]);
        const bool lg = floatAct(actions_.squeeze[0]), rg = floatAct(actions_.squeeze[1]);
        const bool lc = boolAct(actions_.stickClick[0]), rc = boolAct(actions_.stickClick[1]);
        const bool ba = boolAct(actions_.btnA), bb = boolAct(actions_.btnB);
        const bool bx = boolAct(actions_.btnX), by = boolAct(actions_.btnY), bm = boolAct(actions_.btnMenu);
        const bool lp = floatAct(actions_.trackpadForce[0]), rp = floatAct(actions_.trackpadForce[1]);
        const bool lf = floatAct(actions_.squeezeForce[0]), rf = floatAct(actions_.squeezeForce[1]);
        const uint32_t mask = (uint32_t)ls | (rs << 1) | (lt << 2) | (rt << 3) | (lg << 4) |
                              (rg << 5) | (lc << 6) | (rc << 7) | (ba << 8) | (bb << 9) |
                              (bx << 10) | (by << 11) | (bm << 12) | (lp << 13) | (rp << 14) |
                              (lf << 15) | (rf << 16);
        if (mask == lastCensusMask_) return;
        lastCensusMask_ = mask;
        LOGI("[xr] input sources bound: L(stick %d trig %d grip %d click %d) "
             "R(stick %d trig %d grip %d click %d) A %d B %d X %d Y %d menu %d | "
             "trackpad L %d R %d, grip force L %d R %d | profile L=\"%s\" R=\"%s\"%s",
             ls, lt, lg, lc, rs, rt, rg, rc, ba, bb, bx, by, bm, lp, rp, lf, rf,
             ProfileFor("/user/hand/left").c_str(), ProfileFor("/user/hand/right").c_str(),
             (!ls || !rs) ? "  <-- A THUMBSTICK HAS NO BOUND SOURCE (menu/movement will be dead)" : "");
        if (lf || rf)
            LOGI("[xr] Valve Index: the grips read their squeeze force (full at %.2f, [Buttons] IndexGripForce), "
                 "the trackpads press as LeftPad=%s / RightPad=%s", Cfg().indexGripForce,
                 Cfg().btnLeftPad.c_str(), Cfg().btnRightPad.c_str());
    }

    bool LocateIn(XrSpace space, int64_t displayTimeNs, VREyeView out[2]) {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (!bound_ || space == XR_NULL_HANDLE) return false;
        XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
        li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        li.displayTime = displayTimeNs;
        li.space = space;
        XrViewState vs{XR_TYPE_VIEW_STATE};
        XrView views[2]{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
        uint32_t n = 0;
        XrResult r = xrLocateViews(session_, &li, &vs, 2, &n, views);
        if (XR_FAILED(r) || n < 2) return false;
        const bool ok = (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) &&
                        (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT);
        for (int i = 0; i < 2; ++i) {
            out[i].pose.pos = ToVec(views[i].pose.position);
            out[i].pose.rot = ToQuat(views[i].pose.orientation);
            out[i].pose.valid = ok;
            out[i].fov = ToFov(views[i].fov);
        }
        return ok;
    }

    // ---- setup helpers ----
    template <class T> T GetProc(const char* name) const {
        PFN_xrVoidFunction fn = nullptr;
        if (XR_FAILED(xrGetInstanceProcAddr(instance_, name, &fn))) return nullptr;
        return reinterpret_cast<T>(fn);
    }

    bool CreateInstance() {
        uint32_t n = 0;
        if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr))) return false;
        std::vector<XrExtensionProperties> props(n, {XR_TYPE_EXTENSION_PROPERTIES});
        if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data()))) return false;
        auto has = [&](const char* e) {
            for (auto& p : props) if (strcmp(p.extensionName, e) == 0) return true;
            return false;
        };
        if (!has(XR_KHR_D3D12_ENABLE_EXTENSION_NAME)) {
            static bool warned = false;
            if (!warned) { warned = true; LOGE("[xr] runtime does not support %s", XR_KHR_D3D12_ENABLE_EXTENSION_NAME); }
            return false;
        }
        std::vector<const char*> exts{XR_KHR_D3D12_ENABLE_EXTENSION_NAME};
        haveTimeConv_ = has(XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME);
        if (haveTimeConv_) exts.push_back(XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME);
        haveRefresh_ = has(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
        if (haveRefresh_) exts.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
        if (Cfg().vrXrDebug && has(XR_EXT_DEBUG_UTILS_EXTENSION_NAME)) exts.push_back(XR_EXT_DEBUG_UTILS_EXTENSION_NAME);

        XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
        strcpy_s(ci.applicationInfo.applicationName, "ERVR");
        strcpy_s(ci.applicationInfo.engineName, "ERVR");
        ci.applicationInfo.applicationVersion = 1;
        // Request 1.0, NOT XR_CURRENT_API_VERSION. The 1.1 SDK makes that macro 1.1, and
        // runtimes that implement only OpenXR 1.0 (SteamVR, some VDXR builds) reject a 1.1
        // instance with XR_ERROR_INITIALIZATION_FAILED (-4). d2vr proves 1.0 works here.
        ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        ci.enabledExtensionCount = (uint32_t)exts.size();
        ci.enabledExtensionNames = exts.data();
        XrResult r = xrCreateInstance(&ci, &instance_);
        if (XR_FAILED(r)) {
            static XrResult s_last = XR_SUCCESS;
            if (s_last != r) {
                s_last = r;
                LOGW("[xr] xrCreateInstance failed: %s (%d). The runtime is registered but not accepting "
                     "connections - with Virtual Desktop/VDXR the headset must be CONNECTED AND STREAMING "
                     "before the game starts. Retrying every 2 s.", XrStr(r), (int)r);
            }
            instance_ = XR_NULL_HANDLE;
            return false;
        }
        XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};
        if (XR_SUCCEEDED(xrGetInstanceProperties(instance_, &ip)))
            LOGI("[xr] runtime \"%s\" %u.%u.%u", ip.runtimeName,
                 XR_VERSION_MAJOR(ip.runtimeVersion), XR_VERSION_MINOR(ip.runtimeVersion), XR_VERSION_PATCH(ip.runtimeVersion));
        return true;
    }

    bool GetSystem() {
        XrSystemGetInfo gi{XR_TYPE_SYSTEM_GET_INFO};
        gi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        XrResult r = xrGetSystem(instance_, &gi, &systemId_);
        if (XR_FAILED(r)) { systemId_ = XR_NULL_SYSTEM_ID; return false; }   // headset off: retry later, quietly

        uint32_t n = 0;
        xrEnumerateViewConfigurationViews(instance_, systemId_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &n, nullptr);
        viewConfig_.assign(n, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        XRC(xrEnumerateViewConfigurationViews(instance_, systemId_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                              n, &n, viewConfig_.data()), "xrEnumerateViewConfigurationViews");
        if (viewConfig_.size() < 2) { LOGE("[xr] stereo view config unavailable"); return false; }
        XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
        if (XR_SUCCEEDED(xrGetSystemProperties(instance_, systemId_, &sp))) {
            LOGI("[xr] system \"%s\": recommended eye %ux%u, max %ux%u, up to %u layers", sp.systemName,
                 viewConfig_[0].recommendedImageRectWidth, viewConfig_[0].recommendedImageRectHeight,
                 viewConfig_[0].maxImageRectWidth, viewConfig_[0].maxImageRectHeight, sp.graphicsProperties.maxLayerCount);
            // Projection, cinema, HUD and every overlay can be live at once; EndFrame leaves out what does not fit.
            maxLayers_ = sp.graphicsProperties.maxLayerCount;
            if (maxLayers_ && maxLayers_ < 3u + OverlayCount)
                LOGW("[xr] the runtime composes at most %u layers but ERVR may submit %d: the last overlays (the weapon wheel) can be left out",
                     maxLayers_, 3 + (int)OverlayCount);
        }
        return true;
    }

    bool CreateSpaces() {
        XrReferenceSpaceCreateInfo ci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        ci.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
        ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
        if (XR_FAILED(xrCreateReferenceSpace(session_, &ci, &appSpace_))) {
            ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            XRC(xrCreateReferenceSpace(session_, &ci, &appSpace_), "xrCreateReferenceSpace(LOCAL)");
            LOGI("[xr] reference space: LOCAL (STAGE unavailable)");
        } else {
            LOGI("[xr] reference space: STAGE");
        }
        ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
        XRC(xrCreateReferenceSpace(session_, &ci, &viewSpace_), "xrCreateReferenceSpace(VIEW)");
        return true;
    }

    XrPath Path(const char* s) const {
        XrPath p = XR_NULL_PATH;
        xrStringToPath(instance_, s, &p);
        return p;
    }
    bool MakeAction(XrAction& a, const char* name, const char* localized, XrActionType type) {
        XrActionCreateInfo ci{XR_TYPE_ACTION_CREATE_INFO};
        strcpy_s(ci.actionName, name);
        strcpy_s(ci.localizedActionName, localized);
        ci.actionType = type;
        XRC(xrCreateAction(actions_.set, &ci, &a), name);
        return true;
    }

    bool CreateActions() {
        XrActionSetCreateInfo si{XR_TYPE_ACTION_SET_CREATE_INFO};
        strcpy_s(si.actionSetName, "ervr");
        strcpy_s(si.localizedActionSetName, "ERVR");
        si.priority = 0;
        XRC(xrCreateActionSet(instance_, &si, &actions_.set), "xrCreateActionSet");

        const char* side[2] = {"left", "right"};
        for (int h = 0; h < 2; ++h) {
            char n[64];
            sprintf_s(n, "hand_pose_%s", side[h]);   if (!MakeAction(actions_.pose[h], n, n, XR_ACTION_TYPE_POSE_INPUT)) return false;
            sprintf_s(n, "trigger_%s", side[h]);     if (!MakeAction(actions_.trigger[h], n, n, XR_ACTION_TYPE_FLOAT_INPUT)) return false;
            sprintf_s(n, "squeeze_%s", side[h]);     if (!MakeAction(actions_.squeeze[h], n, n, XR_ACTION_TYPE_FLOAT_INPUT)) return false;
            sprintf_s(n, "stick_%s", side[h]);       if (!MakeAction(actions_.stick[h], n, n, XR_ACTION_TYPE_VECTOR2F_INPUT)) return false;
            sprintf_s(n, "stick_click_%s", side[h]); if (!MakeAction(actions_.stickClick[h], n, n, XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;
            sprintf_s(n, "haptic_%s", side[h]);      if (!MakeAction(actions_.haptic[h], n, n, XR_ACTION_TYPE_VIBRATION_OUTPUT)) return false;
            sprintf_s(n, "trackpad_%s", side[h]);    if (!MakeAction(actions_.trackpad[h], n, n, XR_ACTION_TYPE_VECTOR2F_INPUT)) return false;
            sprintf_s(n, "trackpad_force_%s", side[h]); if (!MakeAction(actions_.trackpadForce[h], n, n, XR_ACTION_TYPE_FLOAT_INPUT)) return false;
            sprintf_s(n, "squeeze_force_%s", side[h]);  if (!MakeAction(actions_.squeezeForce[h], n, n, XR_ACTION_TYPE_FLOAT_INPUT)) return false;
        }
        if (!MakeAction(actions_.btnA, "btn_a", "A", XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;
        if (!MakeAction(actions_.btnB, "btn_b", "B", XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;
        if (!MakeAction(actions_.btnX, "btn_x", "X", XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;
        if (!MakeAction(actions_.btnY, "btn_y", "Y", XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;
        if (!MakeAction(actions_.btnMenu, "btn_menu", "Menu", XR_ACTION_TYPE_BOOLEAN_INPUT)) return false;

        SuggestTouch();
        SuggestIndex();
        SuggestSimple();

        for (int h = 0; h < 2; ++h) {
            XrActionSpaceCreateInfo asi{XR_TYPE_ACTION_SPACE_CREATE_INFO};
            asi.action = actions_.pose[h];
            asi.poseInActionSpace = {{0, 0, 0, 1}, {0, 0, 0}};
            XRC(xrCreateActionSpace(session_, &asi, &actions_.handSpace[h]), "xrCreateActionSpace");
        }
        XrSessionActionSetsAttachInfo ai{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        ai.countActionSets = 1;
        ai.actionSets = &actions_.set;
        XRC(xrAttachSessionActionSets(session_, &ai), "xrAttachSessionActionSets");
        return true;
    }

    void Suggest(const char* profile, std::vector<XrActionSuggestedBinding> b) {
        XrInteractionProfileSuggestedBinding s{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
        s.interactionProfile = Path(profile);
        s.countSuggestedBindings = (uint32_t)b.size();
        s.suggestedBindings = b.data();
        XrResult r = xrSuggestInteractionProfileBindings(instance_, &s);
        if (XR_FAILED(r)) LOGW("[xr] bindings for %s rejected: %s", profile, XrStr(r));
    }
    void SuggestTouch() {
        Suggest("/interaction_profiles/oculus/touch_controller", {
            {actions_.pose[0], Path("/user/hand/left/input/grip/pose")},
            {actions_.pose[1], Path("/user/hand/right/input/grip/pose")},
            {actions_.trigger[0], Path("/user/hand/left/input/trigger/value")},
            {actions_.trigger[1], Path("/user/hand/right/input/trigger/value")},
            {actions_.squeeze[0], Path("/user/hand/left/input/squeeze/value")},
            {actions_.squeeze[1], Path("/user/hand/right/input/squeeze/value")},
            {actions_.stick[0], Path("/user/hand/left/input/thumbstick")},
            {actions_.stick[1], Path("/user/hand/right/input/thumbstick")},
            {actions_.stickClick[0], Path("/user/hand/left/input/thumbstick/click")},
            {actions_.stickClick[1], Path("/user/hand/right/input/thumbstick/click")},
            {actions_.btnX, Path("/user/hand/left/input/x/click")},
            {actions_.btnY, Path("/user/hand/left/input/y/click")},
            {actions_.btnA, Path("/user/hand/right/input/a/click")},
            {actions_.btnB, Path("/user/hand/right/input/b/click")},
            {actions_.btnMenu, Path("/user/hand/left/input/menu/click")},
            {actions_.haptic[0], Path("/user/hand/left/output/haptic")},
            {actions_.haptic[1], Path("/user/hand/right/output/haptic")},
        });
    }
    void SuggestIndex() {
        Suggest("/interaction_profiles/valve/index_controller", {
            {actions_.pose[0], Path("/user/hand/left/input/grip/pose")},
            {actions_.pose[1], Path("/user/hand/right/input/grip/pose")},
            {actions_.trigger[0], Path("/user/hand/left/input/trigger/value")},
            {actions_.trigger[1], Path("/user/hand/right/input/trigger/value")},
            {actions_.squeeze[0], Path("/user/hand/left/input/squeeze/value")},
            {actions_.squeeze[1], Path("/user/hand/right/input/squeeze/value")},
            {actions_.stick[0], Path("/user/hand/left/input/thumbstick")},
            {actions_.stick[1], Path("/user/hand/right/input/thumbstick")},
            {actions_.stickClick[0], Path("/user/hand/left/input/thumbstick/click")},
            {actions_.stickClick[1], Path("/user/hand/right/input/thumbstick/click")},
            {actions_.btnX, Path("/user/hand/left/input/a/click")},
            {actions_.btnY, Path("/user/hand/left/input/b/click")},
            {actions_.btnA, Path("/user/hand/right/input/a/click")},
            {actions_.btnB, Path("/user/hand/right/input/b/click")},
            // No Menu here: the Index's only candidate, system/click, is SteamVR's dashboard button and
            // never reaches an app ("the menu button doesn't work", 2026-11-02 report). The left trackpad
            // press is the menu button instead ([Buttons] LeftPad=MENU in the virtual pad).
            {actions_.trackpad[0], Path("/user/hand/left/input/trackpad")},
            {actions_.trackpad[1], Path("/user/hand/right/input/trackpad")},
            {actions_.trackpadForce[0], Path("/user/hand/left/input/trackpad/force")},
            {actions_.trackpadForce[1], Path("/user/hand/right/input/trackpad/force")},
            {actions_.squeezeForce[0], Path("/user/hand/left/input/squeeze/force")},
            {actions_.squeezeForce[1], Path("/user/hand/right/input/squeeze/force")},
            {actions_.haptic[0], Path("/user/hand/left/output/haptic")},
            {actions_.haptic[1], Path("/user/hand/right/output/haptic")},
        });
    }
    void SuggestSimple() {
        Suggest("/interaction_profiles/khr/simple_controller", {
            {actions_.pose[0], Path("/user/hand/left/input/grip/pose")},
            {actions_.pose[1], Path("/user/hand/right/input/grip/pose")},
            {actions_.btnA, Path("/user/hand/right/input/select/click")},
            {actions_.btnX, Path("/user/hand/left/input/select/click")},
            {actions_.btnMenu, Path("/user/hand/left/input/menu/click")},
            {actions_.haptic[0], Path("/user/hand/left/output/haptic")},
            {actions_.haptic[1], Path("/user/hand/right/output/haptic")},
        });
    }

    // First of `wanted` that the runtime offers; 0 if none.
    int64_t PickFormat(std::vector<int64_t> wanted) const {
        uint32_t n = 0;
        if (XR_FAILED(xrEnumerateSwapchainFormats(session_, 0, &n, nullptr)) || n == 0) return 0;
        std::vector<int64_t> have(n);
        if (XR_FAILED(xrEnumerateSwapchainFormats(session_, n, &n, have.data()))) return 0;
        for (int64_t w : wanted)
            if (w != 0 && std::find(have.begin(), have.end(), w) != have.end()) return w;
        return 0;
    }

    bool CreateSwap(Swap& s, uint32_t w, uint32_t h, int64_t format) {
        XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        ci.format = format;
        ci.sampleCount = 1;
        ci.width = w; ci.height = h;
        ci.faceCount = 1; ci.arraySize = 1; ci.mipCount = 1;
        XRC(xrCreateSwapchain(session_, &ci, &s.handle), "xrCreateSwapchain");
        uint32_t n = 0;
        XRC(xrEnumerateSwapchainImages(s.handle, 0, &n, nullptr), "xrEnumerateSwapchainImages");
        s.images.assign(n, {XR_TYPE_SWAPCHAIN_IMAGE_D3D12_KHR});
        XRC(xrEnumerateSwapchainImages(s.handle, n, &n, (XrSwapchainImageBaseHeader*)s.images.data()),
            "xrEnumerateSwapchainImages");
        s.w = w; s.h = h; s.format = format;
        return true;
    }

    ID3D12Resource* AcquireFrom(Swap& s) {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (s.handle == XR_NULL_HANDLE || s.acquired != UINT32_MAX) return nullptr;
        uint32_t idx = 0;
        XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        const double a0 = QpcMs();
        if (XR_FAILED(xrAcquireSwapchainImage(s.handle, &ai, &idx))) return nullptr;
        XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wi.timeout = 100000000;   // 100 ms; never block the render thread indefinitely
        const XrResult wr = xrWaitSwapchainImage(s.handle, &wi);
        g_clkImage.Add(QpcMs() - a0);
        if (XR_FAILED(wr)) {
            XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            xrReleaseSwapchainImage(s.handle, &ri);
            return nullptr;
        }
        s.acquired = idx;
        return s.images[idx].texture;
    }
    void ReleaseTo(Swap& s) {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        if (s.handle == XR_NULL_HANDLE || s.acquired == UINT32_MAX) return;
        XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(s.handle, &ri);
        s.acquired = UINT32_MAX;
    }

    XrTime SampleTime() {
        XrTime t = displayTime_.load();
        if (t != 0) return t;
        if (haveTimeConv_) {
            auto pfn = GetProc<PFN_xrConvertWin32PerformanceCounterToTimeKHR>("xrConvertWin32PerformanceCounterToTimeKHR");
            LARGE_INTEGER now;
            if (pfn && QueryPerformanceCounter(&now) && XR_SUCCEEDED(pfn(instance_, &now, &t))) return t;
        }
        return 0;
    }

    Pose LocateSpace(XrSpace space, XrTime t) {
        Pose out;
        if (space == XR_NULL_HANDLE || appSpace_ == XR_NULL_HANDLE || t == 0) return out;
        XrSpaceVelocity vel{XR_TYPE_SPACE_VELOCITY};
        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION, &vel};
        if (XR_FAILED(xrLocateSpace(space, appSpace_, t, &loc))) return out;
        if (!(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) return out;
        out.rot = ToQuat(loc.pose.orientation);
        out.pos = ToVec(loc.pose.position);
        if (vel.velocityFlags & XR_SPACE_VELOCITY_LINEAR_VALID_BIT)  out.linVel = ToVec(vel.linearVelocity);
        if (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT) out.angVel = ToVec(vel.angularVelocity);
        out.valid = true;
        return out;
    }

    bool GetBool(XrAction a) {
        if (!a) return false;
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = a;
        XrActionStateBoolean s{XR_TYPE_ACTION_STATE_BOOLEAN};
        return XR_SUCCEEDED(xrGetActionStateBoolean(session_, &gi, &s)) && s.isActive && s.currentState;
    }
    // `bound` (optional) says whether the action has a source at all in the current profile.
    float GetFloat(XrAction a, bool* bound = nullptr) {
        if (bound) *bound = false;
        if (!a) return 0.f;
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = a;
        XrActionStateFloat s{XR_TYPE_ACTION_STATE_FLOAT};
        if (XR_FAILED(xrGetActionStateFloat(session_, &gi, &s)) || !s.isActive) return 0.f;
        if (bound) *bound = true;
        return s.currentState;
    }
    XrVector2f GetVec2(XrAction a) {
        if (!a) return {0, 0};
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = a;
        XrActionStateVector2f s{XR_TYPE_ACTION_STATE_VECTOR2F};
        if (XR_FAILED(xrGetActionStateVector2f(session_, &gi, &s)) || !s.isActive) return {0, 0};
        return s.currentState;
    }

    void PumpEvents() {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        for (;;) {
            XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
            if (xrPollEvent(instance_, &ev) != XR_SUCCESS) break;
            switch (ev.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                auto* e = (XrEventDataSessionStateChanged*)&ev;
                sessionState_ = e->state;
                LOGI("[xr] session state -> %d", (int)e->state);
                if (e->state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    if (XR_SUCCEEDED(xrBeginSession(session_, &bi))) {
                        sessionRunning_ = true;
                        LOGI("[xr] session running");
                        ApplyRefreshRate();
                    }
                } else if (e->state == XR_SESSION_STATE_STOPPING) {
                    xrEndSession(session_);
                    sessionRunning_ = false;
                    frameBegun_ = false;
                } else if (e->state == XR_SESSION_STATE_EXITING || e->state == XR_SESSION_STATE_LOSS_PENDING) {
                    TeardownSession();
                    return;
                }
                break;
            }
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                LOGW("[xr] instance loss pending; tearing down");
                TeardownSession();
                if (instance_ != XR_NULL_HANDLE) { xrDestroyInstance(instance_); instance_ = XR_NULL_HANDLE; }
                systemId_ = XR_NULL_SYSTEM_ID;
                return;
            case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                LOGD("[xr] interaction profile changed");
                break;
            case XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB: {
                auto* e = (XrEventDataDisplayRefreshRateChangedFB*)&ev;
                LOGI("[xr] headset refresh changed: %.0f -> %.0f Hz", e->fromDisplayRefreshRate,
                     e->toDisplayRefreshRate);
                refreshHz_.store(e->toDisplayRefreshRate, std::memory_order_relaxed);
                break;
            }
            default: break;
            }
        }
    }

    // Reads the headset's refresh (the game's frame limiter follows it - see
    // PatchesFollowHeadsetRefresh) and, with [VR] HeadsetRefreshHz=N, requests the supported
    // rate nearest N. Runtimes differ in what they let an app pick: VDXR 1.0.10 lists ONLY its
    // current streaming rate (measured 2026-10-31: "supported: 90"), so under Virtual Desktop
    // the rate is the streamer's Frame rate setting and this only reports it.
    void ApplyRefreshRate() {
        if (!haveRefresh_ || session_ == XR_NULL_HANDLE) return;
        auto enumRates = GetProc<PFN_xrEnumerateDisplayRefreshRatesFB>("xrEnumerateDisplayRefreshRatesFB");
        auto getRate = GetProc<PFN_xrGetDisplayRefreshRateFB>("xrGetDisplayRefreshRateFB");
        auto reqRate = GetProc<PFN_xrRequestDisplayRefreshRateFB>("xrRequestDisplayRefreshRateFB");
        if (!enumRates || !getRate || !reqRate) return;
        float cur = 0.f;
        if (XR_FAILED(getRate(session_, &cur)) || cur < 1.f) return;
        refreshHz_.store(cur, std::memory_order_relaxed);
        uint32_t n = 0;
        if (XR_FAILED(enumRates(session_, 0, &n, nullptr)) || n == 0) return;
        std::vector<float> rates(n);
        if (XR_FAILED(enumRates(session_, n, &n, rates.data())) || n == 0) return;
        rates.resize(n);
        std::string list;
        for (float r : rates) { list += std::to_string((int)(r + 0.5f)); list += ' '; }
        const int want = Cfg().vrHeadsetRefreshHz;
        float pick = cur;
        if (want > 0) {
            float best = 1e9f;
            for (float r : rates) if (std::fabs(r - (float)want) < best) { best = std::fabs(r - (float)want); pick = r; }
        }
        if (std::fabs(pick - cur) < 0.5f) {
            LOGI("[xr] headset refresh %.0f Hz (supported: %s)%s", cur, list.c_str(),
                 want > 0 && std::fabs((float)want - cur) >= 0.5f
                     ? " - HeadsetRefreshHz cannot be honoured; set the rate in the streamer "
                       "(Virtual Desktop: Streaming > Frame rate)" : "");
            return;
        }
        const XrResult r = reqRate(session_, pick);
        if (XR_SUCCEEDED(r))
            LOGI("[xr] headset refresh %.0f Hz -> %.0f Hz requested (HeadsetRefreshHz; supported: %s)",
                 cur, pick, list.c_str());
        else
            LOGW("[xr] headset refresh %.0f -> %.0f Hz request failed: %s - set the rate in the "
                 "streamer instead", cur, pick, XrStr(r));
    }
    float DisplayRefreshHz() const override { return refreshHz_.load(std::memory_order_relaxed); }

    void TeardownSession() {
        std::lock_guard<std::recursive_mutex> g(g_mtx);
        for (auto& e : eyes_) e.Destroy();
        quad_.Destroy();
        hud_.Destroy();
        for (auto& o : overlays_) o.Destroy();
        cinemaAnchor_ = Anchor{};
        hudAnchor_ = Anchor{};
        for (auto& s : actions_.handSpace) if (s) { xrDestroySpace(s); s = XR_NULL_HANDLE; }
        if (actions_.set) { xrDestroyActionSet(actions_.set); actions_ = Actions{}; }
        if (appSpace_ != XR_NULL_HANDLE)  { xrDestroySpace(appSpace_);  appSpace_ = XR_NULL_HANDLE; }
        if (viewSpace_ != XR_NULL_HANDLE) { xrDestroySpace(viewSpace_); viewSpace_ = XR_NULL_HANDLE; }
        if (session_ != XR_NULL_HANDLE)   { xrDestroySession(session_); session_ = XR_NULL_HANDLE; }
        sessionRunning_ = false;
        frameBegun_ = false;
        bound_ = false;
        sessionState_ = XR_SESSION_STATE_UNKNOWN;
        device_ = nullptr; queue_ = nullptr;
        displayTime_.store(0);
    }

    XrInstance  instance_ = XR_NULL_HANDLE;
    XrSystemId  systemId_ = XR_NULL_SYSTEM_ID;
    XrSession   session_ = XR_NULL_HANDLE;
    XrSpace     appSpace_ = XR_NULL_HANDLE, viewSpace_ = XR_NULL_HANDLE;
    XrSessionState sessionState_ = XR_SESSION_STATE_UNKNOWN;
    std::vector<XrViewConfigurationView> viewConfig_;
    Swap        eyes_[2], quad_, hud_, overlays_[OverlayCount];
    uint32_t    maxLayers_ = 0;   // XrSystemGraphicsProperties.maxLayerCount, 0 = unknown
    // The HUD quad keeps its own anchor: it follows on a different (tighter) leash than the
    // cinema screen, and in World lock it must not be dragged along by the cinema quad.
    Anchor      cinemaAnchor_, hudAnchor_;
    int64_t     eyeFormat_ = 0;
    Actions     actions_;
    ID3D12Device* device_ = nullptr;
    ID3D12CommandQueue* queue_ = nullptr;
    bool bound_ = false, sessionRunning_ = false, frameBegun_ = false, haveTimeConv_ = false;
    bool haveRefresh_ = false;   // XR_FB_display_refresh_rate enabled on the instance
    std::atomic<float> refreshHz_{0.f};   // the panel's refresh, 0 = unknown
    uint64_t lastAttach_ = 0;
    double   lastCensus_ = 0;
    uint32_t lastCensusMask_ = 0xFFFFFFFF;   // never equals a real mask, so the first poll logs
    std::atomic<XrTime> displayTime_{0};
    uint64_t pairIndex_ = 0;
};

} // namespace

IVRBackend* CreateOpenXRBackend() { static OpenXRBackend b; return &b; }
}
