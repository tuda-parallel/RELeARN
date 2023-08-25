set(COVERAGE_REPORT_AVAILABLE OFF)
if(NOT WIN32 AND "${ENABLE_COVERAGE}")
  find_program(LCOV lcov)
  if(NOT
     LCOV
     MATCHES
     "-NOTFOUND")
    add_custom_target(
      coverage-collect
      COMMAND
        ${LCOV} --directory ${CMAKE_BINARY_DIR} --capture --output-file
        ${CMAKE_BINARY_DIR}/coverage.info --exclude "*_deps*" --exclude
        "/usr/include/*"
      VERBATIM
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/coverage.info)

    find_program(GENHTML genhtml)
    if(NOT
       GENHTML
       MATCHES
       "-NOTFOUND")
      add_custom_target(
        coverage-report
        COMMAND ${GENHTML} --demangle-cpp -o coverage
                ${CMAKE_BINARY_DIR}/coverage.info
        DEPENDS coverage-collect
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      set(COVERAGE_REPORT_AVAILABLE ON)
    else()
      message("genhtml executable not found (required for coverage-report)")
    endif()
  else()
    message("lcov executable not found (required for coverage-collect)")
  endif()
endif()
