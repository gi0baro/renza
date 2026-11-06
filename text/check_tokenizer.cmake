set(python "${STRATA_PYTHON}")
foreach(required IN ITEMS "${python}" "${GGUF}" "${STRATA_ROOT}/tools/strata_tokenizer.py"
                          "${TOKENIZER_DIR}/vocab.json")
    if(NOT EXISTS "${required}")
        message("SKIP: tokenizer check requires ${required}")
        return()
    endif()
endforeach()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "STRATA_TOKENIZER=${TOKENIZER_DIR}"
            "${python}" "${CMAKE_CURRENT_LIST_DIR}/check_tokenizer.py"
            "${TOKENIZER_BINARY}" "${GGUF}" --strata "${STRATA_ROOT}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0 AND NOT result EQUAL 77)
    message(FATAL_ERROR "tokenizer check failed (${result})")
endif()
