# KiPCBTools
为解决Kicad内置功能缺陷而编写的插件的仓库
随时反馈BUG

| 名称/Name | 介绍/Desc| 实例/Demo | 
| ------- | ------- | ------- | 
| FreeDiffPair | 从单端线路生成差分对 | <img src="https://github.com/user-attachments/assets/4a24ec38-2c23-4e1e-98a1-fdd1b098aaf4" width="300px"> | 
| PathDiffGen | 从图形线段生成差分对| <img src="https://github.com/user-attachments/assets/4bb90d43-c026-4999-95a7-9d7dc0f92ba4" width="300px"> |

| 当前开发进度 | 工具 |
|------- | ------- |
| 发行/重构 | FreeDiffPair使用旧版插件接口开发，入口位于Pcbnew插件按钮，正在重构，未来将提供独立图形界面 |
| 开发中 | PathDiffGen使用新IPC接口开发，Pcbnew中没有入口，需手动运行脚本，未来将提供独立图形界面  |

# FreeDiffPair 
**自由角度差分对生成器  /  Free Angle Differential Pair **

当前存在的BUG：
> 生成差分对后，会在起点(实际上起终点不区分)产生一个长度为1nm(Kicad数据最小值)的线路，  
> 该1nm线路不会对PCB性能产生影响，仅作为第一条和第三条线路中非常短的连接段。
> (该1nm线路 会产生 DRC警告 悬空线路)   
> 但会对再次使用插件生成差分对时产生影响，1nm的长度单位过小无法参与向量计算，    
> 某些情况下插件可以使用但会产生错误的结果，但大部分情况插件会直接抛出错误。     
> 修正的方法为删除1nm线路，然后使用自由角度拖动第一条或第三条线路吸附到另一条上，    
> 以得到一根连续的单端线，满足插件的输入要求。拖动距离很短不会影响PCB性能。    
 
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

# DiffGen
**路径差分对生成器  /  BoardSegment Differential Pair Generator**  

使用方法和要求
> 输入线路的要求：差分对足够平行、折线(一组图形线段)完全连续
>
> 选择 **一对差分对(两根线路)** 作为 **仅间距和线宽参考**  
> 选择一条或多条连续的图形线段，如果不连续会产生视为不同的折线  
> 不同的折线生成的差分对不会相互连接  
> 使用插件 为每一组图形线段生成差分对
