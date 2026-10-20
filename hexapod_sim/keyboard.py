"""独立控制窗口：将按键/按钮转换为运动命令，与MuJoCo内置快捷键隔离。"""
from threading import Lock


class Keyboard:
    def __init__(self, strength=40):
        self._lock = Lock()
        self.strength = strength
        self.direction = (0, 0, 0)
        self.reset_requested = False
        self.quit_requested = False

    def on_key(self, key):
        with self._lock:
            directions = {ord('W'): (1, 0, 0), ord('S'): (-1, 0, 0),
                          ord('A'): (0, 1, 0), ord('D'): (0, -1, 0),
                          ord('Q'): (0, 0, 1), ord('E'): (0, 0, -1)}
            if key in directions:
                self.direction = directions[key]
            elif key == ord(' '):
                self.direction = (0, 0, 0)
            elif key == ord('R'):
                self.direction = (0, 0, 0)
                self.reset_requested = True
            elif key in (ord('='), ord('+')):
                self.strength = min(150, self.strength + 5)
            elif key == ord('-'):
                self.strength = max(10, self.strength - 5)
            elif key == 256:
                self.quit_requested = True

    def poll(self):
        with self._lock:
            result = (tuple(self.strength * d for d in self.direction),
                      self.reset_requested, self.quit_requested, self.strength)
            self.reset_requested = False
            return result


class ControlWindow:
    """Tk窗口由仿真主线程每20ms处理事件，不向MuJoCo窗口注册键盘回调。"""
    def __init__(self, keyboard):
        import tkinter as tk
        from tkinter import ttk
        self.keyboard = keyboard
        self.root = tk.Tk()
        self.root.title('六足机器人控制')
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self.request_quit)
        self.status = tk.StringVar(value='停止：点击此窗口后使用键盘')
        frame = ttk.Frame(self.root, padding=16)
        frame.pack()
        ttk.Label(frame, text='机器人控制（MuJoCo窗口只用于观察）').grid(row=0, column=0, columnspan=3, pady=6)
        for row, column, text, key in (
            (1, 0, 'Q 转向+', 'Q'), (1, 1, 'W 前进', 'W'), (1, 2, 'E 转向−', 'E'),
            (2, 0, 'A 左移', 'A'), (2, 1, '空格 停止', ' '), (2, 2, 'D 右移', 'D'),
            (3, 0, '− 减小强度', '-'), (3, 1, 'S 后退', 'S'), (3, 2, '+ 增加强度', '+'),
            (4, 0, 'R 重置起点', 'R'),
        ):
            ttk.Button(frame, text=text, width=17,
                       command=lambda k=key: keyboard.on_key(ord(k))).grid(row=row, column=column, padx=3, pady=3)
        ttk.Button(frame, text='退出仿真', command=self.request_quit).grid(row=4, column=2)
        ttk.Label(frame, textvariable=self.status).grid(row=5, column=0, columnspan=3, pady=10)
        ttk.Label(frame, text='按一次持续运动；空格停止。\n切换到其他窗口自动停止，Esc退出。').grid(row=6, column=0, columnspan=3)
        self.root.bind('<KeyPress>', self._on_key)
        self.root.bind('<FocusOut>', self._on_focus_out)
        # Handle robot keys before a focused ttk button's default bindings.
        for widget in (frame, *frame.winfo_children()):
            tags = widget.bindtags()
            widget.bindtags((str(self.root), *(tag for tag in tags if tag != str(self.root))))
        self.root.update()
        self.root.lift()
        self.root.focus_force()

    def _on_key(self, event):
        if event.keysym == 'Escape':
            self.request_quit()
        elif event.char:
            char = event.char.upper()
            if char in 'WSADQER +-=':
                self.keyboard.on_key(ord(char))
        return 'break'  # Avoid space also invoking a focused button.

    def _on_focus_out(self, event):
        self.root.after_idle(self._stop_if_unfocused)

    def _stop_if_unfocused(self):
        focus = self.root.focus_displayof()
        if focus is None or focus.winfo_toplevel() != self.root:
            self.keyboard.on_key(ord(' '))

    def request_quit(self):
        self.keyboard.on_key(256)

    def update(self, command, fallen):
        label = '已跌倒，请按R重置' if fallen else f'X/Y/R={command}（127以上步幅饱和）'
        self.status.set(label)
        self.root.update_idletasks()
        self.root.update()
        self._stop_if_unfocused()

    def close(self):
        self.root.destroy()
