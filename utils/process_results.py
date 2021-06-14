import numpy as np


def process_results(results_path, frames_count):
    methods = {}
    with open(results_path) as f:
        lis = [line.split(',') for line in f]
        for el in lis:
            method_name = el[0].strip()
            vid_name = el[1].strip()
            time_took = float(el[2].strip())
            if method_name not in methods.keys():
                methods[method_name] = {}
            if vid_name not in methods[method_name]:
                methods[method_name][vid_name] = []
            methods[method_name][vid_name].append(time_took)
    print('& jedna & dwie & trzy & pięć & siedem & tłum  \\\\')
    print('\\hline')
    for method_name in methods.keys():
        method_name_latex = method_name.split('.')[0].replace('_', '\\_')
        one_person_fps = round(frames_count / np.average(methods[method_name]['one_person.mp4']), 3)
        two_person_fps = round(frames_count / np.average(methods[method_name]['two_persons.mp4']), 3)
        three_person_fps = round(frames_count / np.average(methods[method_name]['three_persons.mp4']), 3)
        five_person_fps = round(frames_count / np.average(methods[method_name]['five_persons.mp4']), 3)
        seven_person_fps = round(frames_count / np.average(methods[method_name]['seven_persons.mp4']), 3)
        crowd_fps = round(frames_count / np.average(methods[method_name]['crowd.mp4']), 3)
        print('{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\'.format(method_name_latex, one_person_fps,
                                                                                     two_person_fps, three_person_fps,
                                                                                     five_person_fps, seven_person_fps,
                                                                                     crowd_fps))
        print('\\hline')


if __name__ == '__main__':
    frames_cnt = 30
    res_path = '../results/test_results/bottom_up.txt'
    process_results(res_path, frames_cnt)
