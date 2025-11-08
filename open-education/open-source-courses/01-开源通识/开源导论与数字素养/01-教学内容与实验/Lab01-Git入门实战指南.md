# Git入门实战指南 (Git4Beginner)

欢迎来到Git的世界！Git是目前世界上最先进的分布式版本控制系统。简单来说，它可以帮你记录文件的每一次改动，让你不仅能随时“后悔”（回退到之前的版本），还能方便地与他人协作。

本指南将带你完成从安装到第一次“提交”的全过程。

## 0. 准备工作

在开始之前，你需要拥有一个 **GitCode** 账号。

  * 如果还没有，请前往 [https://gitcode.com/](https://gitcode.com/) 注册。
  * *提示：注册后请记住你的用户名和注册邮箱，稍后配置 Git 时会用到。*

-----

## 1. 安装Git客户端

首先，我们需要在你的电脑上安装Git命令行工具。请根据你的操作系统选择对应的安装方式。

### Windows 用户

1.  前往Git官网下载页面：[https://git-scm.com/download/win](https://git-scm.com/download/win)
2.  点击 "Click here to download" 下载最新的 64-bit 安装程序。
3.  运行安装程序。**一路点击 "Next"（下一步）使用默认设置即可**。
4.  安装完成后，在桌面空白处右键点击，如果看到 **"Git Bash Here"** 菜单，说明安装成功。

### macOS用户

大多数 macOS 系统已预装 Git。

1.  打开 "终端(Terminal)" 应用（可以通过 Command + 空格搜索 "Terminal"）。
2.  输入 `git --version` 并回车。
3.  如果显示了版本号（如 `git version 2.x.x`），则无需安装。
4.  如果没有，终端会提示你安装 "Xcode Command Line Tools"，点击 "安装" 并按照提示操作即可。
      * *(备选方案：你也可以像 Windows 一样去 [git-scm.com/download/mac](https://git-scm.com/download/mac) 下载安装包)*

### Linux用户

打开终端，根据你的发行版输入安装命令：

  * **Debian/Ubuntu:** `sudo apt-get install git`
  * **Fedora:** `sudo dnf install git`

-----

## 2. 初次运行配置 (必须做！)

安装后，你必须告诉 Git “你是谁”。Git 在每次提交时都会记录这些信息。

打开你的命令行工具（Windows 用户请右键选择 **Git Bash**，Mac/Linux 用户打开**终端**），依次输入以下两行命令（注意替换为你自己的信息）：

```bash
# 设置你的名字 (建议使用英文名或拼音)
git config --global user.name "Your Name"

# 设置你的邮箱 (建议使用你注册 GitCode 的邮箱)
git config --global user.email "your_email@example.com"
```

*验证配置是否成功：*
输入 `git config --list`，确认你刚才输入的信息出现在列表中。

-----

## 3. Fork目标仓库 (在线操作)

我们需要在 GitCode 上“复制”一份练习仓库到你自己的名下。这个操作叫 **Fork**。

1.  打开浏览器，访问目标仓库：[https://gitcode.com/Gitconomy-Research/Git4Beginner](https://gitcode.com/Gitconomy-Research/Git4Beginner)
2.  在页面右上角找到 **Fork** 按钮，点击它。
3.  按照提示，选择将仓库 Fork 到你的个人账号下。
4.  等待几秒钟，页面会自动跳转。此时，请注意浏览器地址栏，URL 应该变成了：
    `https://gitcode.com/<你的用户名>/Git4Beginner`
    *这代表你已经拥有了这个仓库的完全控制权！*

-----

## 4. Clone：将仓库下载到本地

现在，我们要把你云端的这个仓库“克隆”到你的电脑上。

1.  在你的电脑上创建一个专门放代码的文件夹，比如 `D:\Code` 或 `~/Code`。
2.  在刚才 Fork 成功的页面上（你自己的仓库页面），找到绿色的 **“克隆”** 按钮，点击并复制 **HTTPS** 地址。
      * *地址格式类似于：`https://gitcode.com/你的用户名/Git4Beginner.git`*
3.  回到命令行工具 (Git Bash/终端)，使用 `cd` 命令进入你刚才创建的文件夹：

```bash
# 例如进入 D 盘的 Code 文件夹 (Windows)
cd /d/Code
# 或者 (Mac/Linux)
cd ~/Code
```

5.  执行克隆命令 (粘贴你刚才复制的地址)：

```bash
git clone https://gitcode.com/你的用户名/Git4Beginner.git
```

6.  下载完成后，进入仓库目录：

```bash
cd Git4Beginner
```

*恭喜你！你已经准备好开始实验了。*

-----

## 5. 新手基础实验

Git 的核心工作流是一个循环：**修改文件 -\> 添加到暂存区 (Add) -\> 提交存档 (Commit) -\> 推送到云端 (Push)**。

### 实验一：你好，世界 (创建新文件)

我们来创建一个属于你自己的文件并提交。

**Step 1: 创建文件**
在 `Git4Beginner` 文件夹下，创建一个新的文本文件，命名为 `hello_<你的名字>.txt`（例如 `hello_guo.txt`），并在里面写上一句你想说的话，然后保存。

**Step 2: 查看状态 (Status)**
回到命令行，输入：

```bash
git status
```

*你会看到红色的文字，提示有一个 "Untracked file"（未跟踪文件），这就是你刚创建的文件。*

**Step 3: 添加到暂存区 (Add)**
告诉 Git 你想让它管理这个新文件：

```bash
git add .
```

*(注意：`add` 后面有一个空格和一个点 `.`，表示添加当前目录下的所有变动)*

**Step 4: 提交存档 (Commit)**
将暂存区的内容正式存入版本历史，并附上一条说明信息：

```bash
git commit -m "My first commit: created hello file"
```

*如果成功，你会看到 `[master (root-commit) xxxxxxx] My first commit...` 这样的提示。*

**Step 5: 推送到云端 (Push)**
将你本地的存档同步到 GitCode 服务器：

```bash
git push
```

*注意：第一次推送时，Git 会弹窗要求你输入 GitCode 的用户名和密码。输入正确后，推送即可成功。*
*(如果看到 `100%` 和 `Done` 字样，说明推送成功！)*

> **验证：** 此时刷新你的 GitCode 仓库网页，你刚刚创建的文件应该已经出现在网页上了！

-----

### 实验二：修改文件

这次我们修改一个已有的文件。

**Step 1: 修改**
用记事本或编辑器打开你刚才创建的 `hello_<你的名字>.txt`，增加一行新内容并保存。

**Step 2: 再次查看状态**

```bash
git status
```

*这次你会看到提示 `modified: hello_<你的名字>.txt`，告诉你有文件被修改了。*

**Step 3: 提交修改三部曲**
重复我们熟悉的操作：

```bash
git add .
git commit -m "Update: added a new line"
git push
```

-----

### 实验三：查看历史 (Log)

你想知道这个仓库之前发生过什么吗？

在命令行输入：

```bash
git log
```

你将看到一个详细的列表，显示了谁（Author）、在什么时间（Date）、做了什么操作（刚才 `-m` 后面的消息）。

*按 `q` 键可以退出查看模式。*

-----

## 恭喜！

你已经掌握了Git最核心的20%的功能，这足以应对 80%的日常个人开发场景。接下来，你可以尝试在 [`Git4Beginner` ](https://gitcode.com/Gitconomy-Research/Git4Beginner)仓库中进行更多的探索。

---

## 许可声明

本文档采用 [知识共享署名--相同方式共享 4.0 国际许可协议 (CC BY--SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 进行许可， &copy; 2025 Gitconomy Research社区。 
