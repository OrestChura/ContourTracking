import os
import tkinter
import tkinter.messagebox as mb
import shutil as sh

# Код будет проводить операции с текущей директорией!!
root = tkinter.Tk()
ans = mb.askokcancel('Предупреждение', 'Будет проведено переименование файлов в текущей директории.',
                     icon='warning', parent=root)
root.destroy()
if ans:
    files = os.listdir('.')
    i_0 = 1
    i_400 = 1
    for f in files:
        if f[:4] == '2019':
            if f[-7:-5] == '_0':
                sh.copy2(f, 'T'+str(i_0)+'_0.tiff')
                i_0 += 1
            elif f[-9:-5] == '_400' and f[-10] != 'n':
                sh.copy2(f, 'T'+str(i_400)+'_400.tiff')
                i_400 += 1
