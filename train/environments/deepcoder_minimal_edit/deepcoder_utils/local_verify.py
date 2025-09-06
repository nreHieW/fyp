import ast
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Union

from .legacy.taco import (
    CODE_TYPE,
    EXECUTION_RESULTS,
    compile_and_get_func,
    compare_std_results,
    create_temp_file,
    execute_cb_code,
    process_input_output,
    remove_tmp_files,
    synthesize_cb_code,
    synthesize_std_code,
)


def _execute_std_code_local(
    method,
    program_code: str,
    inputs_list: List,
    outputs_list: List,
    timeout: int,
    early_stop: bool = False,
    debug: bool = False,
) -> Dict[int, Tuple[bool, str]]:
    temp_program_path = create_temp_file(program_code)
    if debug:
        print("Test program:", temp_program_path)

    assert isinstance(inputs_list, list) and isinstance(outputs_list, list)
    assert len(inputs_list) == len(outputs_list)

    exec_results: Dict[int, Tuple[bool, str]] = {}
    if debug:
        exec_results["debug"] = {}

    for i, inputs in enumerate(inputs_list):
        remove_tmp_files()
        outputs = outputs_list[i]

        if isinstance(inputs, list):
            inputs = [str(k) for k in inputs]
            inputs = "\n".join(inputs)
        if isinstance(outputs, list):
            outputs = [str(k) for k in outputs]
            outputs = "\n".join(outputs)

        stdout, stderr = "", ""
        try:
            result = subprocess.run(
                [sys.executable, temp_program_path],
                input=inputs,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            stdout, stderr = result.stdout, result.stderr
            return_code = result.returncode
            exec_code = 999
        except subprocess.TimeoutExpired as e:
            if debug:
                print(repr(e))
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            stderr = "TIMEOUT"
            return_code = -9
            exec_code = -1
        except Exception as e:
            if debug:
                print(repr(e))
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
            return_code = -99
            stderr = f"{e}"
            exec_code = -2

        stdout = stdout.rstrip("\n")

        if exec_code > 0:
            if return_code == 0:
                if compare_std_results(stdout, outputs, debug):
                    exec_code = 1
                else:
                    exec_code = 0
            else:
                exec_code = -3

        assert exec_code != -3 or return_code != 0
        exec_results[i] = (
            exec_code == 1,
            EXECUTION_RESULTS[exec_code] if exec_code > -3 else EXECUTION_RESULTS[exec_code].format(code=return_code),
        )

        if early_stop and exec_code <= 0:
            break

    return exec_results


def run_test_local(in_outs, test=None, debug=False, timeout=90):
    test = f"{test}"
    if isinstance(in_outs, str):
        try:
            in_outs = ast.literal_eval(in_outs)
            assert isinstance(in_outs, dict)
        except (ValueError, SyntaxError) as e:
            print(f"run_tests_local, Error parsing string: {e}")
            return []

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs["fn_name"]

    inputs_list: List = []
    outputs_list: List = []
    for index, inputs in enumerate(in_outs["inputs"]):
        outputs = in_outs["outputs"][index]
        inputs, outputs = process_input_output(inputs, outputs)
        inputs_list.append(inputs)
        outputs_list.append(outputs)

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")
    if test is None:
        return None

    results: List = []
    if debug:
        print(f"loading test code = {datetime.now().time()}")

    if which_type == CODE_TYPE.call_based:
        synthesized_code = synthesize_cb_code(test, debug)
        method_func = compile_and_get_func(synthesized_code, which_type, method_name, timeout=timeout, debug=debug)
        if not method_func:
            results.append(-2)
            return results
        detail_results, _ = execute_cb_code(
            method_func,
            inputs_list,
            outputs_list,
            timeout=timeout,
            early_stop=True,
            debug=debug,
        )
    else:
        synthesized_code, exec_code = synthesize_std_code(test, debug)
        method_func = compile_and_get_func(synthesized_code, which_type, method_name, timeout=timeout, debug=debug)
        if not method_func:
            results.append(-2)
            return results
        detail_results = _execute_std_code_local(
            method_func,
            exec_code,
            inputs_list,
            outputs_list,
            timeout=timeout,
            early_stop=True,
            debug=debug,
        )

        detail_results = {k: v for k, v in detail_results.items() if k != "debug"}
        if set(detail_results.values()) == {(False, "returncode:1")}:
            synthesized_code, exec_code = synthesize_std_code(test, debug)
            detail_results = _execute_std_code_local(
                method_func,
                synthesized_code + "\ncode()\n",
                inputs_list,
                outputs_list,
                timeout=timeout,
                early_stop=True,
                debug=debug,
            )

    if isinstance(detail_results, list):
        if len(detail_results) == 1:
            detail_results = detail_results * len(inputs_list)
        detail_results = dict(zip([i for i in range(len(inputs_list))], detail_results))

    for _, test_result in detail_results.items():
        if test_result[1] == "passed":
            results.append(True)
        elif test_result[1] == "false":
            results.append(False)
        elif test_result[1] == "timeout":
            results.append(-1)
        else:
            results.append(-3)

    return results


def check_correctness_local(
    tests: Union[List[Dict[str, str]], Dict[str, List[str]]],
    code: str,
    test_fn,
    timeout_per_test: int = 60,
    max_tests: int = 5,
) -> bool:
    if isinstance(tests, list):
        total_tests = len(tests)
        if total_tests > max_tests:
            selected_indices = sorted(range(total_tests), key=lambda i: len(tests[i]["input"]), reverse=True)[:max_tests]
            tests = [tests[i] for i in selected_indices]
    elif isinstance(tests, dict):
        if "inputs" in tests:
            total_tests = len(tests["inputs"])
            if total_tests > max_tests:
                selected_indices = sorted(range(total_tests), key=lambda i: len(tests["inputs"][i]), reverse=True)[:max_tests]
                selected_tests = {
                    "inputs": [tests["inputs"][i] for i in selected_indices],
                    "outputs": [tests["outputs"][i] for i in selected_indices],
                }
                if "fn_name" in tests:
                    selected_tests["fn_name"] = tests["fn_name"]
                tests = selected_tests

    try:
        result = test_fn(
            tests,
            test=code,
            timeout=timeout_per_test,
        )
        if not result:
            return False
        if isinstance(result, List):
            return all(r is True for r in result)
        else:
            return False
    except Exception as e:
        print(f"Error in check_correctness_local: {repr(e)}")
        return False


def primeintellect_check_correctness_local(
    tests,
    code,
    timeout_per_test=60,
    max_tests: int = 5,
):
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)

    formatted_tests: Dict[str, Union[List, str]] = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        formatted_tests["fn_name"] = fn_name

    return check_correctness_local(
        tests=formatted_tests,
        code=code,
        test_fn=run_test_local,
        timeout_per_test=timeout_per_test,
        max_tests=max_tests,
    )


def verify_deepcoder_local(
    completion: str,
    verification_info: dict,
    timeout_per_test: int = 60,
    max_tests: int = 5,
) -> int:
    model_code = completion
    tests = json.loads(verification_info["ground_truth"])
    dataset_name = verification_info["dataset_type"]

    if tests is None:
        raise ValueError("No test cases found")

    if dataset_name == "primeintellect":
        is_correct = primeintellect_check_correctness_local(
            tests=tests,
            code=model_code,
            timeout_per_test=timeout_per_test,
            max_tests=max_tests,
        )
    elif dataset_name == "taco":
        is_correct = check_correctness_local(
            tests=tests,
            code=model_code,
            test_fn=run_test_local,
            timeout_per_test=timeout_per_test,
            max_tests=max_tests,
        )
    else:
        raise ValueError(f"Test type {dataset_name} is not supported")

    return 1 if is_correct else 0
