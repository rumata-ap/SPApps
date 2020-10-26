# coding: utf8
import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as etree
import numpy as np
import csv

# %%
tree = etree.parse('BetonSelector.xml')

types = tree.findall('Type')
groups = []
damps = tree.findall('Damps')
dampsB = damps[0].attrib['vals'].split(',')
lebs = damps[0].findall('Leb')[0]
lebts = damps[0].findall('Lebt')[0]
lebs = np.array(list(map(float, lebs.attrib['vals'].split(','))))
lebts = np.array(list(map(float, lebts.attrib['vals'].split(','))))
lebs = lebs.reshape(3, 3)*0.001
lebts = lebts.reshape(3, 3)*0.001
typesB = []
for typ in types:
    typesB.append(typ.attrib['name'])
# %%
betons = []
# %%


def getGroups(idxtype: int):
    groupsB = []
    groups = types[idxtype].findall('Group')
    for gr in groups:
        groupsB.append(gr.attrib['name'])

    return groupsB


def getClasses(idxtype: int, idxgroup: int):
    classes = []
    groups = types[idxtype].findall('Group')
    classes = groups[idxgroup].findall('Classes')[0].attrib['vals'].split(',')

    return classes

# %%


class Beton (object):
    def __init__(self, name):
        self.name = name
        self.descr = ''
        self.C = []
        self.CL = []
        self.N = []
        self.NL = []
        self.diagr2C = ()
        self.diagr3C = ()
        self.diagr2CL = ()
        self.diagr3CL = ()
        self.diagr2N = ()
        self.diagr3N = ()
        self.diagr2NL = ()
        self.diagr3NL = ()

    def write(self, filepath: str):
        void=['','','','','','']
        header = ["", "", "C", "CL", "N", "NL"]
        r1 = ["Модуль деформации/упругости", "Eb", self.C[0], self.CL[0], self.N[0], self.NL[0]]
        r2 = ["Прочность бетона при сжатии", "Rb", self.C[1], self.CL[1], self.N[1], self.NL[1]]
        r3 = ["Прочность бетона при растяжении", "Rbt", self.C[2], self.CL[2], self.N[2], self.NL[2]]
        r4 = ["Относительная деформация растяжения", "ebt2", self.C[3], self.CL[3], self.N[3], self.NL[3]]
        r5 = ["Относительная деформация растяжения", "ebt0", self.C[4], self.CL[4], self.N[4], self.NL[4]]
        r6 = ["Относительная деформация растяжения", "ebt1", self.C[5], self.CL[5], self.N[5], self.NL[5]]
        r7 = ["Относительная деформация сжатия", "eb1", self.C[6], self.CL[6], self.N[6], self.NL[6]]
        r8 = ["Относительная деформация сжатия", "eb0", self.C[7], self.CL[7], self.N[7], self.NL[7]]
        r9 = ["Относительная деформация сжатия", "eb2", self.C[8], self.CL[8], self.N[8], self.NL[8]]
        r10 = ["Приведенная деформация растяженя", "ebt1red", self.C[9], self.CL[9], self.N[9], self.NL[9]]
        r11 = ["Приведенная деформация сжатия", "eb1red", self.C[10], self.CL[10], self.N[10], self.NL[10]]

        with open(filepath, "w", newline="") as file:
            src=[self.descr,void,header,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11]
            writer = csv.writer(file, dialect='excel', delimiter=';')
            writer.writerows(src)


# %%

def center(window, dx=0, dy=-150):

    x = (window.winfo_screenwidth() - window.winfo_reqwidth()) / 2 + dx
    y = (window.winfo_screenheight() - window.winfo_reqheight()) / 2 + dy
    
    window.wm_geometry("+%d+%d" % (x, y))


class WinSelector(tk.Tk):
    def __init__(self):
        super().__init__()

        self.resizable(width=False, height=False)
        center(self)

        def calcChars():
            """Функция вычисления характеристик бетона на основе данных формы

            Returns:
                (Beton): объект класса 'Beton'
            """
            tb = self.types.current()
            G = types[tb].findall('Group')
            gb = self.groups.current()

            Ebs = G[gb].findall('Eb')[0]
            Rbs = G[gb].findall('Rb')[0]
            Rbts = G[gb].findall('Rbt')[0]
            Rbns = G[gb].findall('Rbn')[0]
            Rbtns = G[gb].findall('Rbtn')[0]
            Rbtns = G[gb].findall('Rbtn')[0]
            Fib_crs = G[gb].findall('Fib_cr')[0]
            ebs = Ebs.attrib['vals'].split(',')
            rbs = Rbs.attrib['vals'].split(',')
            rbts = Rbts.attrib['vals'].split(',')
            rbns = Rbns.attrib['vals'].split(',')
            rbtns = Rbtns.attrib['vals'].split(',')
            fib_cr = Fib_crs.attrib['vals'].split(',')
            fe = Ebs.attrib['factor']
            fb = Rbs.attrib['factor']
            fbt = Rbts.attrib['factor']
            fbn = Rbns.attrib['factor']
            fbtn = Rbtns.attrib['factor']
            ffib = Fib_crs.attrib['factor']
            sfib = Fib_crs.attrib['dim'].split(',')

            ebs = np.array(list(map(float, ebs)))*float(fe)
            rbs = np.array(list(map(float, rbs)))*float(fb)
            rbts = np.array(list(map(float, rbts)))*float(fbt)
            rbns = np.array(list(map(float, rbns)))*float(fbn)
            rbtns = np.array(list(map(float, rbtns)))*float(fbtn)
            fib_cr = np.array(list(map(float, fib_cr)))*float(ffib)
            fib_cr = fib_cr.reshape(int(sfib[0]), int(sfib[1]))

            cb = self.classes.current()
            db = self.damps.current()
            lb = self.layer.current()
            sb = self.structs.current()
            hb = self.hards.current()

            ke = 1
            if tb == 1 and gb == 0:
                ke = 0.89
            elif tb == 3 and hb == 1:
                ke = 0.8
            # ke = 0.89 if tb == 1 & gb == 0 elif tb == 3 & hb == 1 1
            # ke = 0.8 if tb == 3 & hb == 1 else 1
            ro = float(G[gb].attrib['ro']) if tb == 3 else 2200
            id = int(G[gb].attrib['idx']) if tb == 3 else 0
            kfi = (ro/2200) ** 2

            # Вычисления
            gamma_b2 = 0.9 if sb == 0 else 1
            gamma_b3 = 0.85 if lb == 0 else 1

            # Кратковременные расчетные характеристики
            Eb = ebs[cb]*ke
            Rb = rbs[cb+id]*gamma_b2*gamma_b3
            Rbt = -rbts[cb+id]
            ebt2 = -0.00015
            ebt0 = -0.0001
            ebt1 = 0.6*Rbt/Eb
            eb1 = 0.6*Rb/Eb
            eb0 = 0.002
            eb2 = 0.0035
            ebt1red = -0.000085
            eb1red = 0.0015
            diagr3C = ([ebt2, ebt0, ebt1, 0, eb1, eb0, eb2], 
                       [Rbt, Rbt, 0.6*Rbt, 0, 0.6*Rb, Rb, Rb])
            diagr2C = ([ebt2, ebt1red, 0, eb1red, eb2], [Rbt, Rbt, 0, Rb, Rb])
            C = [Eb, Rb, Rbt, ebt2, ebt0, ebt1, eb1, eb0, eb2, ebt1red, eb1red]

            # Длительные расчетные характеристики
            Ebt = Eb/(1+fib_cr[db, cb+id]*kfi)
            Rb = 0.9*Rb
            Rbt = 0.9*Rbt
            eb1 = 0.6*Rb/Ebt
            eb0 = lebs[db, 0]
            eb2 = lebs[db, 1]
            ebt1 = 0.6*Rbt/Ebt
            ebt0 = -lebts[db, 0]
            ebt2 = -lebts[db, 1]
            eb1red = lebs[db, 2]
            ebt1red = -lebts[db, 2]
            diagr3CL = ([ebt2, ebt0, ebt1, 0, eb1, eb0, eb2],
                        [Rbt, Rbt, 0.6*Rbt, 0, 0.6*Rb, Rb, Rb])
            diagr2CL = ([ebt2, ebt1red, 0, eb1red, eb2],
                        [Rbt, Rbt, 0, Rb, Rb])
            CL = [Ebt, Rb, Rbt, ebt2, ebt0, ebt1,
                  eb1, eb0, eb2, ebt1red, eb1red]

            # Нормативные кратковременные характеристики
            Rb = rbns[cb+id]
            Rbt = -rbtns[cb+id]
            ebt2 = -0.00015
            ebt0 = -0.0001
            ebt1 = 0.6*Rbt/Eb
            eb1 = 0.6*Rb/Eb
            eb0 = 0.002
            eb2 = 0.0035
            ebt1red = -0.000085
            eb1red = 0.0015
            diagr3N = ([ebt2, ebt0, ebt1, 0, eb1, eb0, eb2],
                       [Rbt, Rbt, 0.6*Rbt, 0, 0.6*Rb, Rb, Rb])
            diagr2N = ([ebt2, ebt1red, 0, eb1red, eb2],
                       [Rbt, Rbt, 0, Rb, Rb])
            N = [Eb, Rb, Rbt, ebt2, ebt0, ebt1, eb1, eb0, eb2, ebt1red, eb1red]

            # Нормативные длительные характеристики
            eb1 = 0.6*Rb/Ebt
            eb0 = lebs[db, 0]
            eb2 = lebs[db, 1]
            ebt1 = 0.6*Rbt/Ebt
            ebt0 = -lebts[db, 0]
            ebt2 = -lebts[db, 1]
            eb1red = lebs[db, 2]
            ebt1red = -lebts[db, 2]
            diagr3NL = ([ebt2, ebt0, ebt1, 0, eb1, eb0, eb2],
                        [Rbt, Rbt, 0.6*Rbt, 0, 0.6*Rb, Rb, Rb])
            diagr2NL = ([ebt2, ebt1red, 0, eb1red, eb2],
                        [Rbt, Rbt, 0, Rb, Rb])
            NL = [Ebt, Rb, Rbt, ebt2, ebt0, ebt1,
                  eb1, eb0, eb2, ebt1red, eb1red]

            # Вывод
            beton = Beton(self.classes.get())
            beton.descr = [self.types.get(), self.groups.get(),
                           self.classes.get(), self.damps.get(), self.layer.get(), self.structs.get()]
            # beton.descr = self.types.get()+', '+self.groups.get()+', '+self.classes.get() + \
            #     ', ' + self.damps.get()+', '+self.layer.get()
            beton.C = C
            beton.CL = CL
            beton.N = N
            beton.NL = NL
            beton.diagr2C = diagr2C
            beton.diagr3C = diagr3C
            beton.diagr2CL = diagr2CL
            beton.diagr3CL = diagr3CL
            beton.diagr2N = diagr2N
            beton.diagr3N = diagr3N
            beton.diagr2NL = diagr2NL
            beton.diagr3NL = diagr3NL

            return beton

        self.group_1 = tk.LabelFrame(self, padx=5, pady=5,
                                     text="Начальные параметры", width=800)
        self.group_1.pack(padx=5, pady=5, fill='both')

        tk.Label(self.group_1, text="Тип бетона").grid(row=0)
        tk.Label(self.group_1, text="Влажность").grid(row=1)
        tk.Label(self.group_1, text="Конструкция").grid(row=2)
        tk.Label(self.group_1, text="Бетонирование").grid(row=3)
        self.types = ttk.Combobox(
            self.group_1, values=typesB, state="readonly",)
        self.types.current(0)
        self.types.grid(row=0, column=1, sticky=tk.W)

        self.damps = ttk.Combobox(
            self.group_1, values=dampsB, state="readonly")
        self.damps.current(1)
        self.damps.grid(row=1, column=1, sticky=tk.W)

        self.structs = ttk.Combobox(
            self.group_1, values=['Бетонная', 'Железобетонная'], state="readonly")
        self.structs.current(1)
        self.structs.grid(row=2, column=1, sticky=tk.W)

        self.layer = ttk.Combobox(
            self.group_1, values=['Слой больше 1.5м', 'Слой меньше 1.5м'], state="readonly")
        self.layer.current(1)
        self.layer.grid(row=3, column=1, sticky=tk.W)

        # содержимое 2-го фрэйма
        self.group_2 = tk.LabelFrame(self, padx=5, pady=5,
                                     text="Параметры бетона")
        self.group_2.pack(padx=5, pady=5, fill='both')

        tk.Label(self.group_2, text="Группа").grid(row=0)
        tk.Label(self.group_2, text="Класс").grid(row=1)

        self.groups = ttk.Combobox(self.group_2, values=getGroups(
            self.types.current()), state="readonly")
        self.groups.current(0)
        self.groups.grid(row=0, column=1, padx=[28, 0])

        self.classes = ttk.Combobox(self.group_2, values=getClasses(
            self.types.current(), self.groups.current()), state="readonly")
        self.classes.current(6)
        self.classes.grid(row=1, column=1, padx=[28, 0])

        tk.Label(self.group_2, text="Твердение").grid(row=2)
        self.hards = ttk.Combobox(
            self.group_2, values=['Автоклавное', 'Естественное'], state='disabled')
        self.hards.current(0)
        self.hards.grid(row=2, column=1, padx=[28, 0])

        # привязки функций обратного вызова
        def callbackTypes(event):
            sender = event.widget
            idx = sender.current()
            if idx == 3:
                self.hards.config(state="readonly")
            else:
                self.hards.config(state="disabled")
            self.groups.config(values=getGroups(idx))
            self.groups.current(0)
            self.classes.config(values=getClasses(idx, self.groups.current()))
            self.classes.current(0)
            if idx == 0:
                self.classes.current(6)

            def callbackGroup(event):
                self.classes.config(values=getClasses(
                    self.types.current(), self.groups.current()))
                self.classes.current(0)

            self.groups.bind("<<ComboboxSelected>>", callbackGroup)

        self.types.bind("<<ComboboxSelected>>", callbackTypes)

        def writeBeton():
            betons.append(calcChars())
            betons[len(betons)-1].write('Concrete00'+str(len(betons))+'.csv')
            print(betons[len(betons)-1].descr)
            print('C', betons[len(betons)-1].C)
            print('CL', betons[len(betons)-1].CL)
            print('N', betons[len(betons)-1].N)
            print('NL', betons[len(betons)-1].NL)

        self.btn_submit = tk.Button(self, text="Записать", command=writeBeton)
        self.btn_submit.pack(padx=10, pady=10, side=tk.RIGHT)


if __name__ == "__main__":
    app = WinSelector()
    app.title('Выбор бетона')
    app.mainloop()
