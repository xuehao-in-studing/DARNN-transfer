import concurrent.futures
import subprocess
import subprocess
import time


def run_script(param1, param2):
    process = subprocess.Popen(['python', 'predicts.py',
                                 '--targetdomain', str(param1),
                                 '--object_col', str(param2)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


if __name__ == "__main__":
    # 定义一组不同的参数
    experiment_params = [
        {"targetdomain": "HZW", "object_col": "DJ_5"},
        {"targetdomain": "HZW", "object_col": "DX_DW_1"},
        {"targetdomain": "HZW", "object_col": "DX_DW_2"},
        {"targetdomain": "HZW","object_col": "DX_DS_1"},
        {"targetdomain": "HZW","object_col": "DX_DS_2"},
        # 可以继续添加更多参数组合
    ]
    start_time = time.time()
    processes = []

    for params in experiment_params:
        process = run_script(params["targetdomain"], params["object_col"])
        processes.append(process)
    print("hel")
    for process in processes:
        stdout, stderr = process.communicate()
        print(f"Output of {process.args}:\n{stdout.decode('utf-8')}")
        if stderr:
            print(f"Error in {process.args}:\n{stderr.decode('utf-8')}")
    print(f"experiments finished, {len(experiment_params)} experiments have been run, "
          f"consume {(time.time() - start_time)/60:.2f} minutes.")