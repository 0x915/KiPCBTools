# PathDiffGen

沿所选图形折线(线段)路径生成一对差分走线的 KiCad 10 IPC API 插件。

调用接口与旧版 `pcbnew.ActionPlugin`(如 FreeDiffPair)不同:本插件使用
kicad-python(kipy)IPC API,入口为 `main.py` 中的 `main(client: kipy.kicad.KiCad)`,
由 KiCad 作为外部 Python 进程启动。

## 注册到 KiCad 界面

1. 确保插件目录位于 KiCad 会扫描的任一目录下(KiCad 会**递归**扫描):
   - 用户插件目录:`<KICAD_DOCUMENTS_HOME>/<版本>/plugins`
   - 第三方(PCM)目录:`<KICAD_DOCUMENTS_HOME>/<版本>/3rdparty`
   - 本仓库当前位于 `D:\Documents\KiCad\10.0\3rdparty\...`,已在扫描范围内,无需复制。
2. 首次使用前,在 KiCad **Preferences → Plugins** 中:
   - 勾选 **Enable KiCad API**(必须开启,否则插件不会运行);
   - 确认 **Path to Python interpreter** 指向正确的解释器(默认使用 KiCad 自带 Python)。
3. 重启 KiCad,打开 PCB 编辑器。KiCad 会自动为该插件创建虚拟环境并安装
   `requirements.txt` 中的依赖(`kicad-python`)。
   环境就绪后,工具栏/PCB 编辑器首选项中将出现 **PathDiffGen** 动作按钮。

## 文件说明

| 文件 | 作用 |
| ---- | ---- |
| `plugin.json` | KiCad 插件清单(IPC API v1 schema),声明动作、入口与作用域 |
| `requirements.txt` | 声明 `kicad-python`,由 KiCad 装进插件虚拟环境 |
| `main.py` | 插件入口(算法 + kipy 调用),日志输出到 stdout 与 `plugin.log` |
| `plugin.log` | 运行日志(每次启动覆盖,与 FreeDiffPair 相同模式) |

共享库 `_math` / `_logging` 位于插件目录的父目录(仓库根),由 `main.py` 自动加入
`sys.path`;本地 `site-packages`(含 kipy)仅用于 IDE/独立调试,虚拟环境中以 KiCad
安装的为准。

## 图标(待补充)

`plugin.json` 目前未配置图标。制作好图标后,在 `actions` 中追加(文件放在本目录):

```json
"icons-light": ["icon.png"],
"icons-dark": ["icon.png"]
```

## 独立调试

KiCad 未运行 API 服务器时可直接用解释器运行入口(会报连接失败,属正常):

```powershell
D:\kicad\bin\python.exe main.py
```

要连接已运行中的 KiCad,请确保 `Enable KiCad API` 已开启,并设置
`KICAD_API_SOCKET` / `KICAD_API_TOKEN` 环境变量(或使用 KiCad 生成的临时 socket)。
