# psw

`psw` 基于 [ksw2](https://github.com/lh3/ksw2) 做了定向改造，当前重点是用 SIMD 加速序列比对流程（目前以 DNA 场景为主），并为后续 `profile-profile` / `profile-sequence` 比对打基础。

## 项目目标

- 在经典动态规划比对框架上做工程化与性能优化尝试
- 对比不同实现版本（标量/SIMD）在实际数据上的速度与行为差异
- 保持代码结构可扩展，便于继续做蛋白质与 profile 方向优化

## 快速构建

推荐优先使用 CMake（仓库已提供 `CMakeLists.txt`）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/psw
```

也可尝试 Makefile（包含 `native/lto/avx2` 开关）：

```bash
make
./psw
```

性能相关可选参数示例：

```bash
make native=0
make lto=0
make avx2=1
```

## 主要文件说明

### 根目录核心源码

- `main.c`: 程序入口，参数解析与调用流程组织
- `psw.h`: 主要对外接口与公共定义
- `psw_gg.c`: 基础比对实现（参考版本）
- `psw_gg2.c`: 第二版实现/优化路径
- `psw_gg3.c`: 第三版实现/优化路径
- `psw_gg3_sse.c`: 面向 SSE 的实现或加速分支
- `kalloc.c`, `kalloc.h`: 轻量内存分配器实现与声明
- `kseq.h`: 序列读取相关头文件（来源于 klib 风格接口）

### 构建与脚本

- `CMakeLists.txt`: CMake 构建入口（支持 `PSW_ENABLE_NATIVE`、`PSW_ENABLE_LTO`）
- `Makefile`: 传统构建方式与调参开关
- `run_case.sh`: 用例批量运行脚本
- `run_perf.sh`: 性能测试/对比脚本

### 数据与参考实现

- `test/`: 测试数据（如 `protein-perf-*`, `MT-*`）与 case 目录
- `ksw2/`: 上游/参考实现代码与测试数据，便于对照验证

## 后续可完善方向

- 在 README 中补充输入参数格式与输出字段说明
- 为 DNA 与蛋白质场景分别提供基准命令
- 增加正确性回归测试与性能回归基线
