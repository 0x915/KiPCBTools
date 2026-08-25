# KiPCBTools
为解决Kicad内置功能缺陷而编写的插件的仓库
随时反馈BUG

| 名称/Name | 介绍/Desc| 实例/Demo | 
| ------- | ------- | ------- | 
| FreeDiffPair | 从单端线路生成差分对 | <img src="https://github.com/user-attachments/assets/4a24ec38-2c23-4e1e-98a1-fdd1b098aaf4" width="300px"> | 
| PathDiffGen | 从图形线段生成差分对| <img src="https://github.com/user-attachments/assets/4bb90d43-c026-4999-95a7-9d7dc0f92ba4" width="300px"> |

| 当前开发进度 | 工具 |
|------- | ------- |
| 停止维护 | FreeDiffPair(旧版插件接口)，入口位于Pcbnew插件按钮，未来将重构到IPC接口 |
| 开发中 | PathDiffGen(新版IPC接口)，Pcbnew中没有入口，需手动运行脚本，未来将提供图形界面  |

# FreeDiffPair 
**自由角度差分对生成器  /  Free Angle Differential Pair **

使用方法和要求
> 输入线路的要求：差分对足够平行、单端线完全连续
> 
> 1 (表格A实例)  
> 1 选择 **一对差分对(两根线路)** 作为生成起点  
> 1 选择与 **所选的差分对关联** 的 **连续单端线**  
> 1 使用插件 根据单端线生成缺失的单侧差分线  
>    
> 2 (表格B实例)  
> 2 选择 **两对差分对(四根线路)** 且 **缺失线路都在同一侧**  
> 2 选择与 **所选的两个差分对关联** 的 **连续单端线**  
> 2 使用插件 生成中间缺失的单侧差分线  
>   
> 3 (表格C实例)  
> 3 选择 **两对差分对(四根线路)** 但 **仅其中一侧线路没相互连接**   
> 3 使用插件 将两根单端线连接于交点  
 
| 实例/Demo | 图片/Picture | 
| ------- | ------- |
| A | <img src="https://github.com/user-attachments/assets/8ae0bc44-3d5b-4bde-a8d1-9847c4459814" width="500px"> |
| B | <img src="https://github.com/user-attachments/assets/9d40940b-b2c4-40a0-bb3a-05e4a244b43f" width="500px"> |  
| C | <img src="https://github.com/user-attachments/assets/4112692c-5783-4701-9482-3d8ab21c7b50" height="142px"> <img src="https://github.com/user-attachments/assets/49167115-05db-4ec9-b8ad-3fadbf7d1b13" height="142px"> |

# PathDiffGen
**路径差分对生成器  /  BoardSegment Differential Pair Generator**  

目前运行要求：
> - 单pcbnew运行
> - 在控制台cmd/powershell中cd到插件目录PathDiffGen下
> - 使用python解释器(已安装kicad-python依赖)直接运行main.py   
> 命令 python main.py   
> 如果不满足输入要求控制台日志会打印错误信息   
> 安装依赖 python -m pip install kicad-python   


使用方法和要求
> 输入线路的要求：差分对足够平行、折线(一组图形线段)完全连续
>
> 选择 **一对差分对(两根线路)** 作为 **仅间距和线宽参考**  
> 选择一条或多条连续的图形线段，如果不连续会产生视为不同的折线  
> 不同的折线生成的差分对不会相互连接  
> 使用插件 为每一组图形线段生成差分对
