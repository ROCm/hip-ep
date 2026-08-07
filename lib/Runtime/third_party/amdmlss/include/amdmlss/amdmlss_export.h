
#ifndef MLSS_API_H
#define MLSS_API_H

#ifdef MLSS_STATIC_DEFINE
#  define MLSS_API
#  define MLSS_NO_EXPORT
#else
#  ifndef MLSS_API
#    ifdef c_api_EXPORTS
        /* We are building this library */
#      define MLSS_API __declspec(dllexport)
#    else
        /* We are using this library */
#      define MLSS_API __declspec(dllimport)
#    endif
#  endif

#  ifndef MLSS_NO_EXPORT
#    define MLSS_NO_EXPORT 
#  endif
#endif

#ifndef MLSS_DEPRECATED
#  define MLSS_DEPRECATED __declspec(deprecated)
#endif

#ifndef MLSS_DEPRECATED_EXPORT
#  define MLSS_DEPRECATED_EXPORT MLSS_API MLSS_DEPRECATED
#endif

#ifndef MLSS_DEPRECATED_NO_EXPORT
#  define MLSS_DEPRECATED_NO_EXPORT MLSS_NO_EXPORT MLSS_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef MLSS_NO_DEPRECATED
#    define MLSS_NO_DEPRECATED
#  endif
#endif

#endif /* MLSS_API_H */
