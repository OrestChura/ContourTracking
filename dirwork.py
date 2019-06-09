import os
import tkinter
import tkinter.messagebox as mb
import shutil as sh

# Код будет проводить операции с текущей директорией!!
def callback():
    s = var.get()
    ans = mb.askokcancel('Предупреждение',
                         'Будет проведено переименование файлов в текущей директории расположения файла dirwork.py.',
                         icon='warning', parent=root)
    if ans:
        files = os.listdir('.')
        direc =  os.getcwd()+'\\s'
        os.mkdir(direc)
        i_0 = 1
        i_400 = 1
        for f in files:
            if  not (f[-3:] == 'pkl'):
                if f[-7:-5] == '_0':
                    sh.copy2(f, s + str(i_0) + '_0.tiff')
                    sh.move(s + str(i_0) + '_0.tiff', direc)
                    i_0 += 1
                elif f[-9:-5] == '_400' and f[-10] != 'n':
                    sh.copy2(f, s + str(i_400) + '_400.tiff')
                    sh.move(s + str(i_400) + '_400.tiff', direc)
                    i_400 += 1
                else: pass
            else: pass

        root.destroy()

root = tkinter.Tk()
root.title('dirwork')

message = tkinter.Entry(root)
message.insert(0, 'Введите имя:')
message.config(state='readonly')
message.focus_set()
message.pack(pady=10, padx=10)

var = tkinter.StringVar()
textbox = tkinter.Entry(root, textvariable=var)
textbox.focus_set()
textbox.pack(pady=10, padx=10)

listbox = tkinter.Listbox(root, selectmode=tkinter.MULTIPLE)
listbox.pack()
for item in ["400", "660"]:
    listbox.insert(tkinter.END, item)

b = tkinter.Button(root, text="OK", command=callback)
b.pack()

root.mainloop()