# coding:utf-8
import cv2
import numpy as np
import os,sys,time
from hxnumocr.numrec import OneHotFullLinkNetwork as NumberNet
from colorama import Fore, Back, Style
from collections import deque
import hashlib
from PIL import Image as PilImage
from PIL import ImageGrab as PilImageGrab
import win32clipboard

r'''
1. 读取图片
2. 划分网格
3. 识别数字
4. 查找解法集合
'''

def get_number_grids_from_image(file_name:str, dir:str = "imgs", narrow_rate_single:float = 0.15): 
    r'''
    从截屏文件中读取划分好的数字网格，每个网格是numpy格式的单色图片,输出格式是一个tuple，里面三个元素，第一个元素是原始图像np.ndarray,第二个是列表是rect列表，第三个也是列表是网格图片列表([rect,...],[nparray,...])
    '''
    full_file_name = dir + os.path.sep + file_name
    if os.path.exists(full_file_name) == False:
        raise FileNotFoundError(f"未找到图片 {full_file_name}")

    image = cv2.imread(full_file_name, cv2.IMREAD_GRAYSCALE)
    image_colored = cv2.imread(full_file_name)
    # 二值化处理
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # 检测所有轮廓
    all_contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 寻找类似方形的轮廓：即其外切四边形面积超过原面积不到15%，且长宽差距不超过10%的
    # 每个元素是一个外接四边形信息, (x,y,w,h)
    out_rects = np.array([cv2.boundingRect(x) for x in all_contours])

    narrow_rate_single = narrow_rate_single

    rect_to_contour = lambda rect: np.array([[rect[0],rect[1]],[rect[0] + rect[2], rect[1]],[rect[0] + rect[2], rect[1] + rect[3]],[rect[0], rect[1] + rect[3]]])
    get_less_rect = lambda rect: [int(rect[0] + (rect[2] * narrow_rate_single)), int(rect[1] + (rect[3] * narrow_rate_single)), int(rect[2] * (1-narrow_rate_single*2)), int(rect[3] * (1-narrow_rate_single*2))]

    #类方形轮廓组： 0:contour, 1:外接rect (x,y,w,h), 2:外接rect的contour, 3:contour面积，4:rect面积, 5:缩小rect, 6:缩小rect的contour, 7:缩小rect面积
    rect_like_lists = []
    for i in range(len(all_contours)):
        rect_area = out_rects[i][2] * out_rects[i][3]
        contour_area = cv2.contourArea(all_contours[i])
        
        if np.abs(rect_area - contour_area) / contour_area < 0.15 and np.abs(out_rects[i][2] - out_rects[i][3]) / np.min(out_rects[i][2:,]) < 0.1:
            less_rect = get_less_rect(out_rects[i])
            rect_like_lists.append([all_contours[i], out_rects[i], rect_to_contour(out_rects[i]), contour_area, rect_area, less_rect, rect_to_contour(less_rect), less_rect[2] * less_rect[3]])

    # 按照轮廓内面积排序
    rect_like_lists.sort(key = lambda x: x[3])

    if len(rect_like_lists) < 160:
        raise ValueError(f"可选方块数小于160！")

    # 找到连续160个面积差最小的序列，作为方块集合
    best_st = 0
    min_loss = None
    for i in range(0, len(rect_like_lists) - 159):
        loss = np.abs(rect_like_lists[i][3] - rect_like_lists[i + 159][3])
        # print(f"loss at {i}: {loss}")
        if min_loss is None or loss < min_loss:
            best_st = i
            min_loss = loss

    # print(f"best_st at {best_st}, min_loss:{min_loss}")

    # 理论上，select_lists就是160个数字的位置
    select_lists = rect_like_lists[best_st: best_st + 160]

    # 给元素按照棋盘排布排序
    rows, cols = 16, 10
    # 首先按y坐标排序，分出行
    select_lists.sort(key = lambda x: x[5][1])

    sorted_selected_list = []
    for i in range(rows):
        sorted_selected_list.extend(sorted(select_lists[i * cols: (i+1) * cols], key=lambda x: x[5][0]))

    # 绘制轮廓和序号，确认
    img_with_contours = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # 将灰度图转换为彩色图
    cv2.drawContours(img_with_contours, [x[6] for x in select_lists], -1, (0, 255, 0), 2)

    for i in range(len(sorted_selected_list)):
        rect = sorted_selected_list[i][5]
        center_x = rect[0] + rect[2] // 2
        center_y = rect[1] + rect[3] // 2
        cv2.putText(img_with_contours, f"{i + 1}", (center_x - 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0,0,255),2)

    cv2.imwrite(f"temp/grid_check_{file_name.replace('.','_')}.jpg", img_with_contours)
    
    number_core_rects = [grid_item[5] for grid_item in sorted_selected_list]
    number_imgs = [binary[grid_item[5][1]:grid_item[5][1] + grid_item[5][3], grid_item[5][0]:grid_item[5][0] + grid_item[5][2]] for grid_item in sorted_selected_list]
    return (image_colored, number_core_rects, number_imgs)

def build_matrix_from_grid_imgs(number_imgs:list, network_config_dir = "hxnumocr/kd_config"):
    r'''
    从识别出来的小图像生成矩阵
    '''
    numbers = []
    number_network = NumberNet.load_config_and_init_predict_network(network_config_dir)
    for i in range(len(number_imgs)):
        number = int(number_network.predict_single_img(number_imgs[i], True))
        numbers.append(number)

    matrix = np.array(numbers, dtype=np.int8).reshape((16,10))
    return matrix

def print_rect_with_color_terminal(matrix, rect:tuple):
    in_rect = lambda i,j: i >= rect[0] and i < rect[0] + rect[2] and j >= rect[1] and j < rect[1] + rect[3]
    out_str = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print_value = " " if matrix[i][j] == 0 else matrix[i][j]
            if in_rect(i, j):
                out_str += f"{Back.RED}{print_value}{' ' + Style.RESET_ALL if j < rect[1] + rect[3] - 1 else Style.RESET_ALL + ' '}"
            else:
                out_str += f"{print_value} "
        out_str += "\n"
    
    print(out_str)

def rect_sum(matrix, rect:tuple):
    sub_matrix = matrix[rect[0]:rect[0] + rect[2], rect[1]: rect[1] + rect[3]]
    return np.sum(sub_matrix)

def execute_rect(matrix, rect):
    y, x, h, w = rect
    matrix[y : y + h, x : x + w] = 0

    return matrix



def find_all_matches(matrix):
    rows, cols = len(matrix), len(matrix[0])
    matched_rects = []

    for i in range(rows):
        for j in range(cols):
            height_max = False
            for h in range(1, rows - i + 1):
                for w in range(1, cols - j + 1):
                    if h == 1 and w == 1:
                        continue
                    rect = (i,j,h,w)
                    now_sum = rect_sum(matrix, rect)
                    if now_sum == 10:
                        matched_rects.append(rect)
                    if now_sum >= 10:
                        if w == 1:
                            height_max = True
                        break
                if height_max == True:
                    break
    return matched_rects

def find_all_seq_matches(origin_matrix:np.ndarray, shuffle_index:bool = True):
    r'''
    此方法和上一方法的区别在于，此方法寻找的是在不重叠情况下， 一次可以并行消除的所有区域，这样虽然压缩了可能性，但是能有效剪枝
    '''
    matrix = np.copy(origin_matrix)
    rows, cols = len(matrix), len(matrix[0])
    matched_rects = []

    rows_list = list(range(rows))
    cols_list = list(range(cols))

    if shuffle_index:
        np.random.shuffle(rows_list)
        np.random.shuffle(cols_list)
    
    for i in rows_list:
        for j in cols_list:
            # 已经被消除的单元格，跳过检索
            if matrix[i][j] == 0:
                continue
            # 此处需要决定是先纵向扩展还是先横向扩展
            h_first = np.random.rand() < 0.5

            # 先纵向扩展的情况
            if h_first == True:
                # 确认是否达到最大高度的flag
                height_max = False
                for h in range(-i, rows - i + 1):
                    # 行列延申时也判断一下
                    if matrix[i][j] == 0:
                        continue
                    for w in range(-j, cols - j + 1):
                        # 一个单元格一定不会成立，直接继续
                        if (h == 1 and w == 1):
                            continue

                        # 当前搜索的矩形范围及其和
                        rect = (i,j,h,w)
                        now_sum = rect_sum(matrix, rect)

                        # 找到匹配，记载
                        if now_sum == 10:
                            matched_rects.append(rect)
                            execute_rect(matrix, rect)
                        
                        # 找到匹配或超出，宽度遍历结束，break，如果此时宽度为1，那么高度遍历也结束
                        if now_sum >= 10:
                            if w == 1:
                                height_max = True
                            break
                    if height_max == True:
                        break
            else:
                # 先横向扩展的情况

                # 确认是否达到最大宽度的Flag
                width_max = False
                for w in range(-j, cols - j + 1):
                    # 行列延申时也判断一下
                    if matrix[i][j] == 0:
                        continue
                    for h in range(-i, rows - i + 1):
                        # 一个单元格一定不会成立，直接继续
                        if (h == 1 and w == 1):
                            continue

                        # 当前搜索的矩形范围及其和
                        rect = (i,j,h,w)
                        now_sum = rect_sum(matrix, rect)

                        # 找到匹配，记载
                        if now_sum == 10:
                            matched_rects.append(rect)
                            execute_rect(matrix, rect)
                        
                        # 找到匹配或超出，宽度遍历结束，break，如果此时宽度为1，那么高度遍历也结束
                        if now_sum >= 10:
                            if h == 1:
                                width_max = True
                            break
                    if width_max == True:
                        break

    return matched_rects

# steps结构 [current_matrix, rects, score]
class Solution:
    def __init__(self, origin_matrix:np.ndarray):
        self.current_matrix = np.copy(origin_matrix)
        self.rects = []
        self.score = 0
        self.max_score = self.current_matrix.size

    def add_step(self, rect):
        self.rects.append(rect)
        y, x, h, w = rect
        self.score += np.count_nonzero(self.current_matrix[y : y + h, x : x + w])
        execute_rect(self.current_matrix, rect=rect)

    def is_win(self):
        return self.score == self.max_score

    def clone(self):
        new_step = Solution(self.current_matrix)
        new_step.rects = [rect for rect in self.rects]
        new_step.score = self.score
        return new_step
    
    def get_hash(self):
        return hashlib.md5(self.current_matrix.tobytes()).hexdigest()

def find_best_solution(matrix:np.ndarray, get_branchs = lambda: 2, limit_use_rects:int = None, random_search:bool = True, processing_silent:bool = True):
    origin_solution = Solution(matrix)
    solutions_queue = deque([origin_solution])

    best_solution = origin_solution
    calc_count = 0

    matrix_hashs = set()
    matrix_hashs.add(origin_solution.get_hash())
    drops = 0

    while len(solutions_queue) > 0:
        
        solution = solutions_queue.pop()
        
        next_rects_list = [find_all_seq_matches(solution.current_matrix, random_search) for _ in range(get_branchs())]
        for rects in next_rects_list:
            calc_count += 1
            if len(rects) == 0:
                continue
            new_solution = solution.clone()
            
            # 如果设置了主动减除一次操作序列的后半部分
            if limit_use_rects != None and np.random.rand() < 0.5:
                if limit_use_rects < 1:
                    limit_use_rects = max(1, int(len(rects) * limit_use_rects))
                rects = rects[:limit_use_rects]
            for rect in rects:
                new_solution.add_step(rect)
            if new_solution.is_win():
                best_solution = new_solution
                break
            if new_solution.score > best_solution.score:
                best_solution = new_solution
                if not processing_silent:
                    print(f"Calc {calc_count} times, Now Score:{best_solution.score}, Steps:{len(best_solution.rects)} QueueSize:{len(solutions_queue)}")
            # print(new_solution.score, end=' ')
            new_hash = new_solution.get_hash()
            if new_hash in matrix_hashs:
                drops += 1
                # print(f"Drops:{drops}\r", end='')
            else:
                solutions_queue.appendleft(new_solution)
                matrix_hashs.add(new_hash)
                if not processing_silent:
                    print(f"QSize:{len(solutions_queue)}\r", end='')
        
        if best_solution.is_win():
            print(f"Win Solution Found. {best_solution.score} points with {len(best_solution.rects)} Steps")
            break
    if best_solution.is_win() == False:
        print(f"Only Reach:{best_solution.score} points with {len(best_solution.rects)} Steps")
    if not processing_silent:
        print(f"Calc {calc_count} times, Now Score:{best_solution.score}, Steps:{len(best_solution.rects)} QueueSize:{len(solutions_queue)}\n")

    return best_solution

# 在终端中输出步骤
def print_steps_terminal(solution:Solution, origin_matrix:np.ndarray, auto_start:bool = False, auto_interval_ms:float = 0):
    print(f"Best Score: {solution.score}")
    if not auto_start and auto_interval_ms <= 0:
        cmd = input("Press to show step, q to exit: ")
        if cmd == "q":
            return

    display_matrix = np.copy(origin_matrix)
    for i, rect in enumerate(solution.rects):
        os.system("cls")
        print(f"Goal:{solution.score}, {i + 1} / {len(solution.rects)}")
        print_rect_with_color_terminal(display_matrix, rect)
        execute_rect(display_matrix, rect)
        if auto_interval_ms <= 0:
            cmd = input("Press for next, q to exit: ").strip().lower()
            if cmd == "q":
                return
        else:
            time.sleep(auto_interval_ms / 1000)
    print(f"{Back.GREEN}ALL Finished!{Style.RESET_ALL}")

# 生成显示步骤用的图像集合
def gen_steps_imgs(origin_img:np.ndarray, solution:Solution, number_core_rects:list, narrow_rate_single:float) -> tuple:
    r'''
    返回值为tuple(原始图像拷贝, [步骤图像], 结尾图像)
    '''
    # 原始图像拷贝作为基底
    origin_cp = origin_img.copy()
    
    # 存放每一步图像
    step_imgs = []

    # 计算为填补出血，每边朝每一侧应该增加的边长,暂定增加到原尺寸后，向外增加百分之5
    number_img_width = number_core_rects[0][0][2]
    n_rate_recover = int((number_img_width) / (1 - narrow_rate_single * 2) * (0.05 + narrow_rate_single))

    erased_last_step_img = origin_cp.copy()
    for i, step in enumerate(solution.rects):
        # 本次步骤中，左上角的数字方块坐标(y,x,h,w)
        left_top_rect = number_core_rects[step[0]][step[1]]
        # 本步骤中，右下角的数字方块坐标
        right_bottom_rect = number_core_rects[step[0] + step[2] - 1][step[1] + step[3] - 1]
        # 本次操作矩形区域的左上和右下的坐标(y,x, y2,x2)
        lt_y,lt_x,rb_y,rb_x = *left_top_rect[:2], right_bottom_rect[0] + right_bottom_rect[2], right_bottom_rect[1] + right_bottom_rect[3]

        # 预备画红框的图像，需要是上一步擦除好的图像的拷贝
        red_rect_temp = erased_last_step_img.copy()

        # 绘制指示本步骤操作的红框
        red_rect_temp = cv2.rectangle(red_rect_temp,(lt_y - n_rate_recover, lt_x - n_rate_recover), (rb_y + n_rate_recover, rb_x + n_rate_recover), (0,0,255), thickness=number_img_width // 4)
        step_imgs.append(red_rect_temp)

        # 绘制总分提示和步骤提示
        text_pos = (number_core_rects[0][0][0] // 2, 10)
        cv2.putText(red_rect_temp, f"目标分数：{solution.score}, 步骤：{i + 1}/{len(solution.rects)}",text_pos[::-1], cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2)

        # 绘制本步抹消之后的图像
        erased_last_step_img = cv2.rectangle(erased_last_step_img,(lt_y - n_rate_recover, lt_x - n_rate_recover), (rb_y + n_rate_recover, rb_x + n_rate_recover), (88,122,61), thickness=-1)

    # 绘制end图像
    cv2.putText(erased_last_step_img, f"目标分数：{solution.score}, 步骤：{i + 1}/{len(solution.rects)}",text_pos[::-1], cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2)
    cv2.putText(erased_last_step_img, "您已完成所有匹配！", (5, len(erased_last_step_img) // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

    return (origin_cp, step_imgs, erased_last_step_img)

def clear_clipboard():
    r'''
    清空剪贴板
    '''
    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.CloseClipboard()
    except Exception:
        pass
    finally:
        try:
            win32clipboard.CloseClipboard
        except Exception:
            pass


# 使用opencv的图像显示输出步骤
def print_steps_cv2(start_img:np.ndarray, step_imgs:list, end_img:np.ndarray, auto_interval_ms:int = 0, auto_start:bool = False, scale:float = 1):
    auto_interval_ms = max(0, auto_interval_ms)

    show_title = "steps"

    img_show = cv2.resize(start_img, (int(len(start_img[0]) * scale), int(len(start_img) * scale)))
    cv2.imshow(show_title, img_show)
    cv2.waitKey(1000 if auto_start else 0)

    for step_img in step_imgs:
        img_show = cv2.resize(step_img, (int(len(step_img[0]) * scale),int(len(step_img) * scale)))
        cv2.imshow(show_title, img_show)
        cv2.waitKey(auto_interval_ms)
    
    img_show = cv2.resize(end_img, (int(len(end_img[0]) * scale), int(len(end_img) * scale)))
    cv2.imshow(show_title, img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用QT的悬浮半透明窗口显示输出步骤
def print_steps_hover(start_img:np.ndarray, step_imgs:list, end_img:np.ndarray, auto_interval_ms:int = 0, auto_start:bool = False):
    pass
    

def main():
    auto_interval = True
    user_disp_method = input("展示方式(t（默认）:终端,c:OpenCV窗口,h:悬浮窗口):").strip().lower()
    if user_disp_method not in ["c","t","h"]:
        user_disp_method = "t"

    auto_start = input("自动开始展示步骤吗？(y(default)/n):").strip().lower()
    if auto_start == "y" or auto_start == "":
        auto_start = True
    else:
        auto_start = False

    if os.path.exists("temp") == False:
        os.mkdir("temp")
        
    # 从参数中读取文件名的情况
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        full_file_name = sys.argv[1]
        dir = full_file_name[:full_file_name.rfind(os.sep)]
        file_name = full_file_name[full_file_name.rfind(os.sep) + 1:]
        auto_interval = True
    else:
        # 等待输入文件名的情况，如不输入文件名后缀，则默认是imgs文件夹下的.png文件，如输入，则视为完整路径进行检索(相对路径或绝对路径)
        dir = "imgs"

        # 直接回车，则是等待剪贴板
        file_name = input("file name *.png, Enter to wait Clipboard: ").strip()
        if file_name == "":
            pass
        elif "." not in file_name:
            full_file_name = f"{dir}{os.path.sep}{file_name}.png"
        else:
            full_file_name = file_name
            if full_file_name.startswith('"'):
                full_file_name = full_file_name[1:-1]
            dir = full_file_name[:full_file_name.rfind(os.sep)]
            file_name = full_file_name[full_file_name.rfind(os.sep) + 1:]
        auto_interval = True

    # 如果有文件名，则等待剪贴板，没有文件名则等待文件
    if file_name != "":
        print("等待输入的文件存在...")
        while True:
            if os.path.exists(full_file_name) == False:
                time.sleep(0.3)
            else:
                break
    else:
        print("等待剪贴板中的截图中...")
        dir = "temp"
        file_name = f"clipboardimage_{str(int(time.time()))}.png"
        clear_clipboard()
        pil_img = None
        while True:
            pil_img = PilImageGrab.grabclipboard()
            if isinstance(pil_img, PilImage.Image):
                break
            time.sleep(0.3)
        print("等候到剪贴板截图")
        cv_img = np.array(pil_img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{dir}{os.path.sep}{file_name}", cv_img)


    # 记录图像识别花费的时间
    time_st = time.time()
    
    # 单个数字取图像时对边缘的padding
    narrow_rate_single = 0.15

    # 从原始图像中分割出按顺序排列的矩形区域、图像数字
    origin_img, number_core_rects, number_imgs = get_number_grids_from_image(file_name,dir,narrow_rate_single=narrow_rate_single)
    number_core_rects = np.array(number_core_rects).reshape((16,10,4))
    
    # 识别所有图像数字区域的数字，排列成和游戏一致的矩阵
    matrix = build_matrix_from_grid_imgs(number_imgs)
    time_end = time.time()
    time_cost = time_end - time_st
    print(f"Img Recognize : {np.round(time_cost, 2)} s.")
    
    # 记录寻找解集的时间
    time_st = time.time()
    # 随机查找几个解
    best_solutions = [find_best_solution(matrix, lambda:np.random.randint(6,11),None,True) for _ in range(2)]
    time_end = time.time()
    time_cost = time_end - time_st  

    # 将解集排序，寻找分数最高、步数最少的解作为最优解
    best_solutions.sort(key=lambda x: (-x.score, len(x.rects)))
    best_solution = best_solutions[0]

    # 计算自动步骤情况下，每两个步骤的合理间隔
    if auto_interval == True:
        auto_interval = int((120 - time_cost - 12) / len(best_solution.rects) * 1000)
    
    # 如果不是自动开始，这里停一下
    if auto_start == False:
        input("按回车开始展示步骤...")

    if user_disp_method == "t":
        print_steps_terminal(best_solution, matrix, auto_interval_ms=auto_interval)
    else:
        st_img, step_imgs, end_img = gen_steps_imgs(origin_img, best_solution, number_core_rects, narrow_rate_single)
        if user_disp_method == "c":
            print_steps_cv2(st_img, step_imgs, end_img, auto_interval, False)
        elif user_disp_method == "h":
            print_steps_hover()
    # os.remove(full_file_name)


if __name__ == "__main__":
    main()

    
