#pragma once
#include "k200-diag.h"
#include <string>
extern "C" void k200_profile_add(const char *,double,uint64_t);
struct K200Scope {
 std::string name;uint64_t bytes,start;
 K200Scope(std::string n,uint64_t b=0):name(std::move(n)),bytes(b),start(k200_diag_clock(CLOCK_MONOTONIC)){}
 ~K200Scope(){auto end=k200_diag_clock(CLOCK_MONOTONIC);k200_profile_add(name.c_str(),end-start,bytes);if(k200_diag_event)k200_diag_event("audit",name.c_str(),"",-1,1,start,end,0,bytes);}
};
