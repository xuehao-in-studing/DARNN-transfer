import concurrent.futures
import subprocess
import subprocess
import time


def run_script(param1, param2):
    process = subprocess.Popen(['python', 'predicts.py',
                                 '--dataroot', str(param1),
                                 '--object_col', str(param2)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


if __name__ == "__main__":
    # 定义一组不同的参数
    experiment_params = [
        {"dataroot": "../data/东线环切数据_DJ_5.csv", "object_col": "DJ_5"},
        {"dataroot": "../data/东线环切数据_DX_DW_1.csv", "object_col": "DX_DW_1"},
        {"dataroot": "../data/东线环切数据_DX_DW_2.csv", "object_col": "DX_DW_2"},
        {"dataroot": "../data/东线环切数据_DX_DS_1.csv", "object_col": "DX_DS_1"},
        {"dataroot": "../data/东线环切数据_DX_DS_2.csv", "object_col": "DX_DS_2"},
        # 可以继续添加更多参数组合
    ]
    start_time = time.time()
    processes = []

    for params in experiment_params:
        process = run_script(params["dataroot"], params["object_col"])
        processes.append(process)
    print("hel")
    for process in processes:
        stdout, stderr = process.communicate()
        print(f"Output of {process.args}:\n{stdout.decode('utf-8')}")
        if stderr:
            print(f"Error in {process.args}:\n{stderr.decode('utf-8')}")
    print(f"experiments finished, {len(experiment_params)} experiments have been run, "
          f"consume {(time.time() - start_time)/60:.2f} minutes.")