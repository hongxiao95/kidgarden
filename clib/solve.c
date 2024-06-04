#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <Windows.h>

#define HXDLL __declspec(dllexport)
#define SWAP(a,b) {int SWAPtemp=(a);(a)=(b);(b)=SWAPtemp;}

// 编译dll的命令： cl /LD xx.c

unsigned int getseeds(){
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned int)(ts.tv_nsec / 100);
}

// 随机整数范围
int randbetween(int a, int b){
    int rint = rand();
    return a + (int)(((double)rint / (RAND_MAX + 1.0)) * (b - a));
}

int* shuffle(int* array, int len){
    for(int i = 0; i < len; i++){
        int rindex = randbetween(0, len);
        SWAP(array[i], array[rindex]);
    }
    return array;
}

// 初始化range,在该函数内部malloc
void init_range(int** array, int a, int b){
    int len = b - a;
    *array = (int*)malloc(sizeof(int) * len);
    for(int i = 0; i < len; i++){
        (*array)[i] = a + i;
    }
}

// 二维数组的坐标转换为一维数组
int m_index(int i, int j, int col_size){
    return i * col_size + j;
}

// 计算矩形范围内的面积
int sum_rect(int* matrix, int col_size, int* rect){
    int sum = 0;
    int y = rect[0];
    int x = rect[1];
    int h = rect[2];
    int w = rect[3];
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            sum += matrix[m_index(i, j, col_size)];
        }
    }
    return sum;
}

// 执行方块
void execute_rect(int* matrix, int col_size, int* rect){
    int y = rect[0];
    int x = rect[1];
    int h = rect[2];
    int w = rect[3];
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            matrix[m_index(i, j, col_size)] = 0;
        }
    }
}

/**
 * 生成正负交错的range
 * @param side_first <0 负数优先 > 0 正数优先, == 0 随机决定优先
 * @returns 生成的range长度
*/
size_t gen_cross_double_range(int** array, int left_stop, int right_stop, int side_first){
    if (side_first == 0){
        side_first = rand() % 2 == 0 ? -1 : 1;
    }
    size_t length = left_stop + right_stop - 2;

    (*array) = (int*)malloc(sizeof(int) * length);


    int insert_index = 0; 
    for(int i = 1; i < max(left_stop, right_stop); i++){
        if(side_first > 0){
            if(i < right_stop){
                (*array)[insert_index++] = i;
            }
            if(i < left_stop){
                (*array)[insert_index++] = -i;
            }
        }else{
            if(i < left_stop){
                (*array)[insert_index++] = -i;
            }
            if(i < right_stop){
                (*array)[insert_index++] = i;
            }
        }
    }
    return length;
}

/**
 * fix宽高为负数的数组
*/
int* fix_rect(y, x, h, w){
    int* rect = (int*)malloc(sizeof(int) * 4);
    rect[0] = y;
    rect[1] = x;
    rect[2] = h;
    rect[3] = w;

    if(h < 0){
        rect[0]-= -h;
        rect[2] = -h + 1;
    }
    if(w < 0){
        rect[1] -= -w;
        rect[3] = -w + 1;
    }

    return rect;
}

void print_matrix(int* matrix, int rows, int cols, char* msg){
    printf("PRINT_MATRIX [%s]:\n", msg);
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {   
            if(matrix[i * cols + j]){
                printf("%d, ", matrix[i * cols + j]);
            }else{
                printf("??, ");
            }
            
        }
        printf("\n");
    }
    printf("\n");
    
}

int get_cpu_count(){
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_process = (int)sysinfo.dwNumberOfProcessors;
    return num_process;
}

/**
 * 寻找一次可走的最大步数
*/
HXDLL int** find_all_seq_matches(int* origin_matrix, int rows, int cols, int shuffle_index, int max_steps){
    
    srand(getseeds());
    int* matrix = (int*)malloc(sizeof(int) * cols * rows);
    for (size_t i = 0; i < rows * cols; i++)
    {
        matrix[i] = origin_matrix[i];
    }

    // 修正maxsteps
    max_steps = max_steps <= 0 ? 99999 : max_steps;
    bool reach_max_step = false;
    
    // 最大矩形数不超过元素数量的一半
    int rects_max_count = (cols * rows) / 2 + 1;
    // printf("rects max : %d\n", rects_max_count);
    int** matched_rects = (int**)malloc(sizeof(int*) * rects_max_count);
    int picked_rect_count = 0;

    // 准备随机行列索引
    int* row_list = NULL;
    int* col_list = NULL;

    // 初始化range时需要传入地址
    init_range(&row_list, 0 ,rows);
    init_range(&col_list, 0, cols);

    if(shuffle_index != 0){
        shuffle(row_list, rows);
        shuffle(col_list, cols);
    }
    

    for(int _i = 0; _i < rows; _i++){
        int i = row_list[_i];
        int height_max = false;
        int width_max = false;
        for(int _j = 0; _j < cols; _j++){
            int j = col_list[_j];

            if(matrix[m_index(i, j, cols)] == 0){
                continue;
            }
            
            bool h_first = rand() % 2 == 1;

            // 高度优先的情况
            if(h_first == true){
                // 高度达到最大的flag
                height_max = false;
                
                // 生成高度检索范围
                int* h_range = NULL;
                int h_range_len = gen_cross_double_range(&h_range, i + 1, rows - i + 1,0);
                
                for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                    int h = h_range[h_dlt_index];
                    // 当前位置为0忽略
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // 生成宽度检索范围
                    int* w_range = NULL;
                    int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1, 0);
                    for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                        int w = w_range[w_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }
                        int* rect = fix_rect(i, j, h, w);
                        int now_sum = sum_rect(matrix, cols, rect);

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            if(picked_rect_count >= max_steps){
                                reach_max_step = true;
                                break;
                            }
                            execute_rect(matrix, cols, rect);
                            height_max = true;
                            break;
                        }else{
                            free(rect);
                        }

                        if(now_sum >= 10 && (w_dlt_index == w_range_len - 1 || abs(w) < abs(w_range[w_dlt_index + 1]))){
                            if(w == 1 && (h_dlt_index == h_range_len - 1 || abs(h) < abs(h_range[h_dlt_index + 1]))){
                                height_max = true;
                            }
                            break;
                        }
                    }
                    free(w_range);
                    if(height_max || reach_max_step){
                        break;
                    }
                    
                }
                free(h_range);
                if(reach_max_step){
                    break;
                }
            }
            else{
                // 宽度达到最大的flag
                width_max = false;
                
                // 生成宽度检索范围
                int* w_range = NULL;
                int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1,0);

                for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                    int w = w_range[w_dlt_index];
                    // 当前位置为0忽略
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // 生成高度检索范围
                    int* h_range = NULL;
                    int h_range_len = gen_cross_double_range(&h_range, i + 1, rows - i + 1, 0);
                    for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                        int h = h_range[h_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }

                        // printf("pre wh fix\n");
                        int* rect = fix_rect(i, j, h, w);
                        int now_sum = sum_rect(matrix, cols, rect);
                        // printf("wh sum\n");

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            if(picked_rect_count >= max_steps){
                                reach_max_step = true;
                                break;
                            }
                            execute_rect(matrix, cols, rect);
                            width_max = true;
                            break;
                        }else{
                            free(rect);
                        }

                        if(now_sum >= 10 && (h_dlt_index == h_range_len - 1 || abs(h) < abs(h_range[h_dlt_index + 1]))){
                            if(h == 1 && (w_dlt_index == w_range_len - 1 || abs(w) < abs(w_range[w_dlt_index + 1]))){
                                width_max = true;
                            }
                            break;
                        }
                    }
                    free(h_range);
                    if(width_max || reach_max_step){
                        break;
                    }
                }
                free(w_range);
                if(reach_max_step){
                    break;
                }
            }
            if(reach_max_step){
                break;
            }
        }
        if(reach_max_step){
            break;
        }
    }
    free(matrix);
    free(row_list);
    free(col_list);
    int* ends_symbol = (int*)malloc(sizeof(int) * 4);
    ends_symbol[0] = -1;
    ends_symbol[1] = -1;
    ends_symbol[2] = -1;
    ends_symbol[3] = -1;

    matched_rects[picked_rect_count] = ends_symbol;

    return matched_rects;
}

typedef struct {
    int* origin_matrix;
    int rows;
    int cols;
    int shuffle_index;
    int max_steps;
    int** result;
} ProcessData;

/**
 * 包装为单参数的线程内函数
*/
DWORD WINAPI thread_function(LPVOID arg){
    ProcessData* data = (ProcessData*) arg;
    data->result = find_all_seq_matches(data->origin_matrix, data->rows, data->cols, data->shuffle_index, data->max_steps);
    return 0;
} 

int get_allowed_thread_count(int cpu_count){
    if(cpu_count <= 12){
        return cpu_count - 1;
    }else if(cpu_count <= 24){
        return cpu_count  -2;
    }else{
        return cpu_count - 3;
    }
}

/**
 * 多线程寻找解的方案，最大程度利用CPU
 * 线程原则：
 * 若逻辑CPU <= 12个，则可开cpu - 1个线程
 * cpu [13,24],可开cpu - 2个线程
 * 24以上，可开cpu - 3个线程
 * 
 * 若branchs <= 线程数，则直接开branch数量个线程
 * branches > 线程数，则走循环，每次开最多branches个线程
*/
HXDLL int *** find_all_seq_matches_mthread(int* origin_matrix, int rows, int cols, int shuffle_index, int branches, int each_max_steps){
    int*** res_list = (int***)malloc(sizeof(int**) * branches);
    int cpu_count = get_cpu_count();
    int max_threads = get_allowed_thread_count(cpu_count);

    int unhandled_thread_count = branches;

    int total_results_index = 0;

    // printf("Unhandled Threads: %d\n", unhandled_thread_count);
    while(unhandled_thread_count > 0){
        int currrent_thread_count = min(max_threads, unhandled_thread_count);
        unhandled_thread_count -= currrent_thread_count;

        // 本批次线程数组
        HANDLE* threads = (HANDLE*)malloc(sizeof(HANDLE) * currrent_thread_count);

        ProcessData* datas = (ProcessData*)malloc(sizeof(ProcessData) * currrent_thread_count);
        // printf("This time has Threads: %d\n", currrent_thread_count);
        for (size_t i = 0; i < currrent_thread_count; i++)
        {
            //装配datas
            int* copy_matrix = (int*)malloc(sizeof(int) * rows * cols);
            for (size_t j = 0; j < rows * cols; j++)
            {
                copy_matrix[j] = origin_matrix[j];
            }
            
            datas[i].origin_matrix = copy_matrix;
            datas[i].rows = rows;
            datas[i].cols = cols;
            datas[i].shuffle_index = shuffle_index;
            datas[i].max_steps = each_max_steps;
            datas[i].result = NULL;

            threads[i] = CreateThread(NULL, 0, thread_function, &datas[i], 0, NULL);
            if(threads[i] == NULL){
                printf("Create Thread Fail.");
            }

            // int assign_cpu = randbetween(0, cpu_count);

            // DWORD_PTR affinity_mask = (DWORD_PTR) 1 << assign_cpu;
            // SetThreadAffinityMask(threads[i], affinity_mask);
        }
        WaitForMultipleObjects(currrent_thread_count, threads, TRUE, INFINITE);

        // 获取本次的结果
        for (size_t i = 0; i < currrent_thread_count; i++)
        {
            CloseHandle(threads[i]);
            res_list[total_results_index++] = datas[i].result;
            free(datas[i].origin_matrix);
        }
        
        free(threads);
        free(datas);

    }

    return res_list;
}

int main(int argc, char** argv){
    
    int array[160] = {2, 5, 9, 9, 2, 1, 8, 4, 2, 9, 5, 4, 9, 5, 3, 8, 2, 9, 6, 9, 2, 9, 7, 8, 6, 3, 3, 8, 7, 6, 8, 8, 9, 2, 3, 1, 6, 8, 9, 5, 2, 2, 5, 7, 2, 4, 3, 9, 5, 5, 7, 3, 8, 1, 6, 7, 8, 7, 3, 1, 7, 8, 6, 1, 2, 5, 1, 8, 1, 4, 4, 3, 7, 7, 6, 8, 1, 1, 6, 2, 8, 4, 9, 4, 7, 7, 1, 1, 6, 8, 2, 4, 5, 3, 9, 7, 5, 3, 6, 9, 4, 2, 5, 1, 9, 1, 6, 5, 1, 7, 4, 1, 4, 6, 3, 7, 9, 9, 9, 1, 4, 5, 9, 3, 1, 2, 7, 4, 3, 5, 3, 9, 1, 6, 2, 2, 4, 3, 8, 3, 4, 4, 3, 4, 4, 1, 9, 4, 2, 6, 4, 3, 5, 1, 7, 6, 8, 1, 5, 9};
    
    int*** rects_list = NULL;
    int branches = 5000;

    int st = time(0);
    rects_list = find_all_seq_matches_mthread(array, 16, 10, 1, branches, 3);
    int end = time(0);
    printf("All Finish: %d s", end - st);
    // printf("finish");
    for(int rects_index = 0; rects_index < branches; rects_index++){
        // printf("res index: %d\n", rects_index);
        int** rects = rects_list[rects_index];
        int check_value = rects[0][0];
        int index = 0;
        while(check_value != -1){
            // for (size_t i = 0; i < 4; i++)
            // {
            //     // printf("%d,", rects[index][i]);
            // }
            // printf("\n");
            check_value = rects[++index][0];
        }
        // printf("Res Index %d: [%d] Steps\n", rects_index, index);
    }

    free(array);
    free(rects_list);
    

    return 0;
}