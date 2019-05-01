
#include "basic.h"

#include <stdlib.h>
#ifdef USE_OPENMP
#include <omp.h>

  struct _stopwatch {
    double start;
    double current;
  };
#else
#ifdef __WIN32__
#  include <Windows.h>
#  include <MMSystem.h>

  struct _stopwatch {
    DWORD start;
    DWORD current;
  };
#else
#  include <sys/times.h>
#  include <unistd.h>
  struct _stopwatch {
    struct tms start;
    struct tms current;
    real clk_tck;
  };
#endif
#endif

pstopwatch
new_stopwatch()
{
  pstopwatch sw;

  sw = (pstopwatch) malloc(sizeof(stopwatch));
  
#ifndef USE_OPENMP
#ifndef WIN32
  sw->clk_tck = sysconf(_SC_CLK_TCK);
#endif
#endif

  return sw;
}

void
del_stopwatch(pstopwatch sw)
{
  free(sw);
}

void
start_stopwatch(pstopwatch sw)
{
#ifdef USE_OPENMP
  sw->start = omp_get_wtime();
#else
#ifdef WIN32
  sw->start = timeGetTime();
#else
  times(&sw->start);
#endif
#endif
}

real
stop_stopwatch(pstopwatch sw)
{
#ifdef USE_OPENMP
  sw->current = omp_get_wtime();
  return (sw->current - sw->start);
#else
#ifdef WIN32
  sw->current = timeGetTime();

  return (sw->current - sw->start) * 0.001;
#else
  times(&sw->current);

  return (sw->current.tms_utime - sw->start.tms_utime
	  + sw->current.tms_stime - sw->start.tms_stime) / sw->clk_tck;
#endif
#endif
}
