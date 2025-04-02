extern "C"
#if _WIN32
    __declspec(dllexport)
#endif
        const char* say_hello() {
  return "hello, world!";
}
