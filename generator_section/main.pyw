from PyQt5 import QtWidgets,  QtGui
import sys
import RC
import pandas as pd
from PyQt5.QtGui import QPixmap
import numpy as np
import matplotlib.tri as tri

# %%Function of generator
def rectf(H, B, nbi, ns, a, ds):
    path = "Rectangle.csv"
    nbx = int(H//nbi)
    nby = int(B//nbi)
    dx = H/nbx
    dy = B/nby
    nb = nbx*nby
    Abi = np.ones(nb)*dx*dy
    Zbxi = np.zeros(nb)
    Zbyi = np.zeros(nb)
    H0 = H/2-0.5*dx
    H_0 = -H/2+0.5*dx
    B0 = B/2-0.5*dy
    B1 = np.full(nbx, B0)
    B_1 = np.full(nbx, -B0)
    Zbxi = np.resize(np.linspace(H0, H_0, nbx), nb)
    Zbyi = np.resize(np.linspace(B1, B_1, nby), nb)

    # CLIP
    Zbxp1_clip = np.linspace(H/2, -H/2, nbx+1).reshape(1, nbx+1)
    Zbxp2_clip = np.full([1, nby-1], H/2)
    Zbxp_clip = np.hstack([Zbxp1_clip, -Zbxp2_clip, -Zbxp1_clip, Zbxp2_clip])
    Zbyp1_clip = np.full([1, nbx+1], B/2)
    Zbyp2_clip = (np.linspace(B/2, -B/2, nby+1))[1:-1].reshape(1, nby-1)
    Zbyp_clip = np.hstack([-Zbyp1_clip, -Zbyp2_clip, Zbyp1_clip, Zbyp2_clip])
    Zbxyp_clip = np.hstack([np.transpose(Zbxp_clip), np.transpose(Zbyp_clip)])

    clip_a = np.zeros([(nbx+nby)*2, 1])
    clip = np.hstack([Zbxyp_clip, clip_a])

    Cp_r = np.transpose(np.array([Zbxi, Zbyi, Abi]))
    Cp_r = np.concatenate([Cp_r, clip])

    # Генератор координат точек арматуры
    ns[0] = ns[0]+2
    ZsxjH = np.linspace(-H/2+a[0], H/2-a[0], ns[0])
    ZsyjH = np.ones(ns[0])*(-B/2+a[1])
    ZsxjB = np.ones(ns[1])*(-H/2+a[0])
    ZsyjB = np.linspace(-B/2+a[1], B/2-a[1], ns[1])
    ZsxjH = np.hstack([ZsxjH[1:-1], -ZsxjH[1:-1]])
    ZsyjH = np.hstack([ZsyjH[1:-1], -ZsyjH[1:-1]])
    ZsxjB = np.hstack([ZsxjB, -ZsxjB])
    ZsyjB = np.hstack([ZsyjB, -ZsyjB])
    ZsxyjH = (np.vstack([ZsxjH, ZsyjH])).transpose()
    ZsxyjB = (np.vstack([ZsxjB, ZsyjB])).transpose()
    ZsxyjHB = np.vstack([ZsxyjH, ZsxyjB])

    dsH1 = np.ones((ns[0]-2)*2)
    dsH = ds[0]*dsH1
    dsB1 = np.ones((ns[1])*2)
    dsB = ds[1]*dsB1
    dsHB = np.hstack([dsH, dsB])

    index_clip = np.arange(len(Cp_r)-(nbx+nby)*2, len(Cp_r))
    index_clip = pd.DataFrame(np.hstack([index_clip, index_clip[0]]))

    epssp = np.zeros(len(dsHB))
    df = pd.concat([pd.DataFrame(ZsxyjHB), pd.DataFrame(
        dsHB), pd.DataFrame(epssp)], axis=1)
    df_1_2 = pd.concat([pd.DataFrame(Cp_r), index_clip, df],
                       axis=1, join='outer')
    df_1_2.to_csv(path, header=['Zbx', 'Zby', 'Ab', 'Clip', 'Zsx', 'Zsy', 'ds', 'epssp'],
                  index=False, sep=';', mode='w')


def circf(max_radius, min_radius, n_angles, n_radii,
          count_rebars, rs, ds):
    path = "Circle.csv"
    radii = np.linspace(min_radius, max_radius, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles
    x = (radii*np.cos(angles)).flatten()
    y = (radii*np.sin(angles)).flatten()
    triang = tri.Triangulation(x, y)
    Zbxi = x[triang.triangles].mean(axis=1)
    Zbyi = y[triang.triangles].mean(axis=1)
    mask = np.where(Zbxi*Zbxi + Zbyi*Zbyi < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)
    n = len(x)
    pts = np.zeros((n, 2))
    pts[:, 0] = x
    pts[:, 1] = y
    tP = pts[triang.triangles]
    Abi = 0.5*np.abs((tP[:, 0, 0]-tP[:, 2, 0])*(tP[:, 1, 1]-tP[:, 2, 1]) -
                     (tP[:, 1, 0]-tP[:, 2, 0])*(tP[:, 0, 1]-tP[:, 2, 1]))

    # CLIP
    alfa = np.linspace(0, 2*np.pi-2*np.pi/n_angles, n_angles)
    Zbyp_clip = max_radius*np.cos(alfa)
    Zbxp_clip = max_radius*np.sin(alfa)
    Zbxyp_clip = np.transpose([Zbxp_clip, Zbyp_clip])
    clip_a = np.zeros([n_angles, 1])
    clip = np.hstack([Zbxyp_clip, clip_a])
    Cp_r = np.transpose(np.array([Zbxi, Zbyi, Abi]))
    Cp_r = np.concatenate([Cp_r, clip])

    def sse(count_rebars):
        L_Zsxyj = []
        alfa = [np.linspace(0, 2*np.pi-2*np.pi/count_rebars[i],
                            count_rebars[i]) for i in range(len(count_rebars))]
        for i in range(len(alfa)):
            Zsxj = rs[i]*np.cos(alfa[i])
            Zsyj = rs[i]*np.sin(alfa[i])
            Zsxyj = (np.vstack([Zsyj, Zsxj])).transpose()
            L_Zsxyj.extend(Zsxyj)
        L_dsy = []
        for i in range(len(ds)):
            ds1 = np.ones(count_rebars[i])
            dsy = ds[i]*ds1
            L_dsy.extend(dsy)
        epssp = np.zeros(sum(count_rebars))

        index_clip = np.arange(len(Cp_r)-n_angles, len(Cp_r))
        index_clip = pd.DataFrame(np.hstack([index_clip, index_clip[0]]))

        df = pd.concat([pd.DataFrame(L_Zsxyj), pd.DataFrame(
            L_dsy), pd.DataFrame(epssp)], axis=1)
        df_1_2 = pd.concat(
            [pd.DataFrame(Cp_r), index_clip, df], axis=1, join='outer')
        df_1_2.to_csv(path, header=['Zbx', 'Zby', 'Ab', 'Clip', 'Zsx', 'Zsy', 'ds', 'epssp'],
                      index=False, sep=';', mode='w')
    sse(count_rebars)


def tavrf(Hpolk, Bpolk, Hrebr, Brebr, nbi, ns, a, ds):
    # Вычисляем координаты центра тяжести относительно нижнего края полки тавра
    Xc = (Hrebr*Brebr*(Hrebr/2+Hpolk) + Hpolk *
          Bpolk*Hpolk/2)/(Hpolk*Bpolk+Hrebr*Brebr)
    # вычисления для полки
    nbpx = int(Hpolk//nbi)
    nbpy = int(Bpolk//nbi)
    nbrx = int(Hrebr//nbi)
    nbry = int(Brebr//nbi)
    dxp = Hpolk/nbpx
    dyp = Bpolk/nbpy
    nbp = nbpx*nbpy
    Abip = np.ones(nbp)*dxp*dyp
    Zbxip = np.zeros(nbp)
    Zbyip = np.zeros(nbp)
    H0p = Hpolk/2-0.5*dxp
    H_0p = -Hpolk/2+0.5*dxp
    B0p = Bpolk/2-0.5*dyp
    B1p = np.full(nbpx, B0p)
    B_1p = np.full(nbpx, -B0p)
    Zbxip = np.resize(np.linspace(H0p, H_0p, nbpx), nbp)
    Zbxip_n = np.array([Xc-Hpolk/2, ])
    Zbxip -= Zbxip_n
    Zbyip = np.resize(np.linspace(B1p, B_1p, nbpy), nbp)
    Zbp = np.transpose(np.array([Zbxip, Zbyip, Abip]))
    # CLIP
    Zbiyp_clip = np.concatenate([np.ones(nbpx-1)*-Bpolk/2, np.linspace(-Bpolk/2, -Brebr/2, nbpy//2-1),
                                 np.ones(nbrx-1)*-Brebr/2, np.linspace(-Brebr/2, Brebr/2, nbry+1)])
    Zbjyp_clip = np.concatenate([np.ones(nbrx-1)*Brebr/2, np.linspace(Brebr/2, Bpolk/2, nbpy//2-1),
                                 np.ones(nbpx-1)*Bpolk/2, np.linspace(Bpolk/2, -Bpolk/2, nbpy+1)])
    Zbyp_clip = np.concatenate([Zbiyp_clip, Zbjyp_clip])
    Zbixp_clip = np.concatenate([np.linspace(Xc, Xc-Hpolk, nbpx+1), np.ones(nbpy//2-3)*Xc -
                                 Hpolk, np.linspace(Xc-Hpolk, -Hpolk-Hrebr+Xc, nbrx+1), np.ones(nbry-1)*-Hpolk-Hrebr+Xc])
    Zbjxp_clip = np.concatenate([np.linspace(-Hpolk-Hrebr+Xc, Xc-Hpolk, nbrx+1), np.ones(
        nbpy//2-3)*Xc-Hpolk, np.linspace(Xc-Hpolk, Xc, nbpx+1), np.ones(nbpy-1)*Xc])
    Zbxp_clip = np.hstack([Zbixp_clip[1:], Zbjxp_clip, Zbixp_clip[0]])
    Zbxyp_clip = np.concatenate(
        [-Zbxp_clip[:, np.newaxis], Zbyp_clip[:, np.newaxis]], axis=1)
    clip_a = np.zeros([(nbpx-1+nbpy//2-1+nbrx-1)*2+nbry+1+nbpy+1, 1])
    clip = np.hstack([Zbxyp_clip, clip_a])
    # вычисления для ребра
    dxr = Hrebr/nbrx
    dyr = Brebr/nbry
    nbr = nbrx*nbry
    Abir = np.ones(nbr)*dxr*dyr
    Zbxir = np.zeros(nbr)
    Zbyir = np.zeros(nbr)
    H0r = Hrebr/2-0.5*dxr
    H_0r = -Hrebr/2+0.5*dxr
    B0r = Brebr/2-0.5*dyr
    B1r = np.full(nbrx, B0r)
    B_1r = np.full(nbrx, -B0r)
    Zbxir = np.resize(np.linspace(H0r, H_0r, nbrx), nbr)
    Zbxir_n = np.array([Xc-Hrebr/2-Hpolk, ])
    Zbxir -= Zbxir_n
    Zbyir = np.resize(np.linspace(B1r, B_1r, nbry), nbr)
    Zbr = np.transpose(np.array([Zbxir, Zbyir, Abir]))

    Cp_r = np.concatenate([Zbp, Zbr, clip])

    # Запись файла объединение данных различной размерности
    path = "Tavr.csv"
    # Генератор координат точек арматуры
    ZsyjPolk = (np.linspace(-Bpolk/2+a[0], Bpolk/2-a[0], int(ns[0])))
    ZsxjPolk = (np.ones(int(ns[0])))*(-Xc+a[0])
    ZsxyjPolk = (np.vstack([ZsxjPolk, ZsyjPolk])).transpose()

    ZsyjRebr = (np.linspace(-Brebr/2+a[1], Brebr/2-a[1], int(ns[1])))
    ZsxjRebr = (np.ones(int(ns[1])))*-(Zbxir_n-(Hrebr/2-a[1]))
    ZsxyjRebr = (np.vstack([ZsxjRebr, ZsyjRebr])).transpose()

    ZsxyjHB = np.vstack([ZsxyjPolk, ZsxyjRebr])

    dsH1 = np.ones(int(ns[0]))
    dsH = ds[0]*dsH1
    dsB1 = np.ones(int(ns[1]))
    dsB = ds[1]*dsB1
    dsHB = np.hstack([dsH, dsB])

    index_clip = np.arange(
        len(Cp_r)-((nbpx-1+nbpy//2-1+nbrx-1)*2+nbry+1+nbpy+1), len(Cp_r))
    index_clip = pd.DataFrame(np.hstack([index_clip, index_clip[0]]))

    epssp = np.zeros(len(dsHB))
    df = pd.concat([pd.DataFrame(ZsxyjHB), pd.DataFrame(
        dsHB), pd.DataFrame(epssp)], axis=1)
    df_1_2 = pd.concat([pd.DataFrame(Cp_r), index_clip, df],
                       axis=1, join='outer')
    df_1_2.to_csv(path, header=['Zbx', 'Zby', 'Ab', 'Clip', 'Zsx', 'Zsy', 'ds', 'epssp'],
                  index=False, sep=';', mode='w')


def slabf(h, k, asj, ds, alpha, step):
    path = "Slab.csv"
    Zs = pd.DataFrame([h/2-asj[0], h/2-asj[0]-(ds[0]+ds[1]) /
                       2, -(h/2-asj[1]-(ds[2]+ds[3])/2), -(h/2-asj[1])])
    epssp = pd.DataFrame(np.zeros(4))
    hk = pd.DataFrame(np.array([h, k]).reshape(1, 2))
    alpha = pd.DataFrame(alpha)
    ds = pd.DataFrame(ds)
    step = pd.DataFrame(step)

    df = pd.concat([hk, Zs, ds, alpha, step, epssp], axis=1)
    df.to_csv(path, header=['h', 'K', 'Zs', 'ds', 'alpha',
                            'step', 'epssp'], index=False, sep=';', mode='w')


# %%
class RCApp(QtWidgets.QMainWindow, RC.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.nsjcir.hide()
        self.nsjcir_lb.hide()
        self.Rcir.hide()
        self.Rcir_lb.hide()
        self.nbRcir.hide()
        self.nbRcir_lb.hide()
        self.hpolk.hide()
        self.hpolk_lbl.hide()
        self.bpolk.hide()
        self.bpolk_lb.hide()
        self.hrebr.hide()
        self.hrebr_lb.hide()
        self.brebr.hide()
        self.brebr_lb.hide()
        self.nbrcir.hide()
        self.nbrcir_lb.hide()
        self.rsjcir.hide()
        self.rsjcir_lb.hide()
        self.dsjcir.hide()
        self.dsjcir_lb.hide()
        self.hslb.hide()
        self.hslb_lb.hide()
        self.kslb.hide()
        self.kslb_lb.hide()
        self.asslb.hide()
        self.asslb_lb.hide()
        self.dsjslb.hide()
        self.dsjslb_lb.hide()
        self.alpha.hide()
        self.alpha_lb.hide()
        self.stepslb.hide()
        self.stepslb_lb.hide()
        self.ns1ns2tvr.hide()
        self.ns1ns2tvr_lb.hide()
        self.recpic.setPixmap(QPixmap('Rec.png'))
        self.tavr.toggled.connect(lambda: self.btnstate(self.tavr))
        self.circ.toggled.connect(lambda: self.btnstate(self.circ))
        self.rect.toggled.connect(lambda: self.btnstate(self.rect))
        self.shell.toggled.connect(lambda: self.btnstate(self.shell))
# %%Data
        # read and set data
        data = pd.read_csv('data.csv', sep=';', index_col=0)
#        conc=pd.read_csv('Concrete.csv', sep=';',index_col=0)
#        reinf=pd.read_csv('Reinf_steel.csv', sep=';',index_col=0)

        self.comboBox_1.addItems(pd.read_csv(
            'Concrete.csv', sep=';', index_col=0).keys())
        self.comboBox_2.addItems(['>75', '40-75', '<40'])
        self.comboBox_1.setCurrentText(data['Value']['Concrete'])
        self.comboBox_2.setCurrentText(data['Value']['Humidity'])
#        self.radioButton_9.setChecked(True)
        if data['Value']['Type_Concrete'] == 'heavy':
            self.radioButton_3.setChecked(True)
        else:
            self.radioButton_4.setChecked(True)
        self.lineEdit_5.setText(data['Value']['gb_3'])
        self.lineEdit_8.setText(data['Value']['Poisson_ratio'])
        # Data reinforcement
        self.comboBox_3.addItems(pd.read_csv(
            'Reinf_steel.csv', sep=';', index_col=0).index[0:6])
        self.comboBox_3.setCurrentText(data['Value']['Rebars'])
        self.comboBox_4.addItems(pd.read_csv(
            'Reinf_steel.csv', sep=';', index_col=0).index[5:18])
        self.comboBox_4.setCurrentText(data['Value']['Tendons'])

        if data['Value']['Duration'] == 'short':
            self.radioButton_7.setChecked(True)
        else:
            self.radioButton_8.setChecked(True)

        if data['Value']['Limits_state'] == '1_st':
            self.radioButton_9.setChecked(True)
        else:
            self.radioButton_10.setChecked(True)

        self.pushButton_1.clicked.connect(lambda: self.on_click())
        self.pushButton_1.clicked.connect(lambda: self.to_csv())


# %%Functions

    def btnstate(self, b):
        if b.text() == 'Circle':
            if b.isChecked() == True:
                self.recpic.setPixmap(QPixmap('Cir.png'))
        elif b.text() == 'Tavr':
            if b.isChecked() == True:
                self.recpic.setPixmap(QPixmap('Tavr.png'))
        elif b.text() == 'Rectangle':
            if b.isChecked() == True:
                self.recpic.setPixmap(QPixmap('Rec.png'))
        elif b.text() == 'Slab':
            if b.isChecked() == True:
                self.recpic.setPixmap(QPixmap('Slab.png'))

    def on_click(self):
        elem = np.array(['Rectangle', 'Circle', 'Tavr', 'Slab'])
        msk = np.array([self.rect.isChecked(), self.circ.isChecked(),
                        self.tavr.isChecked(), self.shell.isChecked()])
        S = elem[msk][0]

        if S == 'Circle':
            R = float(self.Rcir.text())
            r = 0
            nbR = int(self.nbRcir.text())
            nbr = int(self.nbrcir.text())
            ns = self.nsjcir.text()
            rs = self.rsjcir.text()
            ds = self.dsjcir.text()
            ns = [int(i) for i in ns.split(',')]
            rs = [float(i) for i in rs.split(',')]
            ds = [float(i) for i in ds.split(',')]
            circf(R, r, nbR, nbr, ns, rs, ds)
        elif S == 'Rectangle':
            H = float(self.Hrec.text())
            B = float(self.Brec.text())
            nbi = float(self.dlt_nbrec.text())
            ns = self.ns1ns2rec.text()
            a = self.a1a2rec.text()
            ds = self.ds1ds2rec.text()
            ns = [int(i) for i in ns.split(',')]
            a = [float(i) for i in a.split(',')]
            ds = [float(i) for i in ds.split(',')]
            rectf(H, B, nbi, ns, a, ds)
        elif S == 'Tavr':
            Hpolk = float(self.hpolk.text())
            Bpolk = float(self.bpolk.text())
            Hrebr = float(self.hrebr.text())
            Brebr = float(self.brebr.text())
            nbi = float(self.dlt_nbrec.text())
            ns = self.ns1ns2tvr.text()
            a = self.a1a2rec.text()
            ds = self.ds1ds2rec.text()
            ns = [int(i) for i in ns.split(',')]
            a = [float(i) for i in a.split(',')]
            ds = [float(i) for i in ds.split(',')]
            tavrf(Hpolk, Bpolk, Hrebr, Brebr, nbi, ns, a, ds)

        elif S == 'Slab':
            h = float(self.hslb.text())
            k = int(self.kslb.text())
            asj = self.asslb.text()
            ds = self.dsjslb.text()
            alpha = self.alpha.text()
            step = self.stepslb.text()
            asj = [float(i) for i in asj.split(',')]
            ds = [float(i) for i in ds.split(',')]
            alpha = [float(i) for i in alpha.split(',')]
            step = [float(i) for i in step.split(',')]
            slabf(h, k, asj, ds, alpha, step)

    def to_csv(self):
        elem = np.array(['Rectangle', 'Circle', 'Tavr', 'Slab'])
        msk = np.array([self.rect.isChecked(), self.circ.isChecked(),
                        self.tavr.isChecked(), self.shell.isChecked()])
        S = elem[msk][0]
        Grade = self.comboBox_1.currentText()
        Hum = self.comboBox_2.currentText()
        Reb = self.comboBox_3.currentText()
        Tend = self.comboBox_4.currentText()
        if self.radioButton_7.isChecked() == True:
            Dur = 'short'
        else:
            Dur = 'long'

        if self.radioButton_3.isChecked() == True:
            Dens = "heavy"
        else:
            Dens = "light"
        if self.radioButton_9.isChecked() == True:
            State = '1_st'
        else:
            State = '2_st'
        Pois = self.lineEdit_8.text()
        gb_3 = self.lineEdit_5.text()
        data = pd.read_csv('data.csv', sep=';', index_col=0)
        data.loc['Element', 'Value'] = S
        data.loc['Humidity', 'Value'] = Hum
        data.loc['Concrete', 'Value'] = Grade
        data.loc['Type_Concrete', 'Value'] = Dens
        data.loc['Rebars', 'Value'] = Reb
        data.loc['Tendons', 'Value'] = Tend
        data.loc['Duration', 'Value'] = Dur
        data.loc['Limits_state', 'Value'] = State
        data.loc['Poisson_ratio', 'Value'] = Pois
        data.loc['gb_3', 'Value'] = gb_3
        data.to_csv('data.csv', sep=';')


# %%
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RCApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
