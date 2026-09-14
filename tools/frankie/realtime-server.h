#pragma once
#include "runtime-options.h"
#include <memory>
#include <string>
#include <vector>

namespace httplib { class Server; }
class brain_session;
class mouth_session;

// Install the same duplex protocol on a host server. Authentication belongs to
// that host's pre-routing policy; brain and mouth must outlive its listener.
class frankie_realtime_routes {
    struct state;
    std::shared_ptr<state> state_;
  public:
    frankie_realtime_routes(httplib::Server &, brain_session &, mouth_session &,
                           std::string package, frankie_options options);
    bool occupied() const;
};

std::string frankie_reference_transcript(brain_session &, const std::vector<float> & pcm24);
