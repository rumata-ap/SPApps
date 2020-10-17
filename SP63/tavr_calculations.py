import pandas as pd
import numpy as np
d_reb = pd.read_csv('Reinf_steel.csv', sep=';', index_col=0)
d_con = pd.read_csv('Concrete.csv', sep=';', index_col=0)


class Concrete:
    def __init__(self,Con):
        self.Con = Con

    def Rb_n(self):
        Rb_n = -(d_con[self.Con]['Rbn_heavy'])
        return Rb_n

    def Rbt_n(self):
        Rbt_n = d_con[self.Con]['Rbtn_heavy']
        return Rbt_n

    def Eb(self):
        Eb = d_con[self.Con]['Eb_heavy']
        return Eb


class Rebars:
    def __init__(self,clss):
        self.clss = clss

    def E_s(self):
        Es = d_reb['Es'][self.clss]
        return Es


    def Rs_n(self):
        Rsn = d_reb['Rsn'][self.clss]
        return Rsn


    def Rsc_long(self):
        Rsc_long = -(d_reb['Rsc_long'][self.clss])
        return Rsc_long


    def Rsc_short(self):
        Rsc_short = -(d_reb['Rsc_short'][self.clss])
        return Rsc_short

# когда написано property это дает возможность сразу обращаться trt.R_b вместо trt.R_b()
    @property
    def alfa_R(self):
        alfa_r = d_reb['alfa_R'][self.clss]
        return alfa_r


class Tavr(Concrete, Rebars):
    def __init__(self,Con,clss):
        Concrete.__init__(self, Con)
        Rebars.__init__(self, clss)

    def proverka(self,b,bf,h,hf,a,asc,Mfact,As,Asc):
        self.h = h
        self.hf = hf
        self.b = b
        self.bf = bf
        self.a = a
        self.asc = asc
        self.Mfact = Mfact
        self.As = As
        self.Asc = Asc
        Rb = Concrete.Rb_n(self)/1.3
        h0 = h-a
        Rs = Rebars.Rs_n(self)/1.15
        Rsc_short = Rebars.Rsc_short(self)

        if Rs*As <= Rb*bf*hf+Rsc_short*Asc:
            x = (Rs*As-Rsc_short*Asc)/(Rb*bf) # сжатая зона бетона, м
            Mult = Rb*b*x*(h0-0.5*x) + Rsc_short*Asc*(h0-asc)
            if Mfact < Mult:
                return(f"Прочность сечения обеспечена: {round(Mult,3)} > {Mfact}")
            else:
                return("Прочность не обеспечена")
        else:
            x = (Rs*As - Rsc_short*Asc-Rb*(bf-b)*hf)/(Rb*b)
            Mult = Rb*b*x*(h0-0.5*x)+Rb*(bf-b)*hf*(h0-0.5*hf)+Rsc_short*Asc*(h0-asc)
            if Mfact < Mult:
                return(f"Прочность сечения обеспечена: {round(Mult,3)} > {Mfact}")
            else:
                return("Прочность сечения не обеспечена")


    def podbor(self,b,bf,h,hf,a,asc,Mfact,yb1):
        self.h = h
        self.hf = hf
        self.b = b
        self.bf = bf
        self.a = a
        self.yb1 = yb1
        self.asc = asc
        self.Mfact = Mfact
        Rb = yb1*Concrete.Rb_n(self)/1.3
        h0 = h-a
        Rs = Rebars.Rs_n(self)/1.15
        Es = Rebars.E_s(self)
        # Rsc_short = Rebars.Rsc_short(self)
        Er = 0.8/(1+(Rs/Es/0.0035))
        alfa_r = Er*(1-0.5*Er)
        if Mfact <= Rb*bf*hf*(h0-0.5*hf):
            print('В полке')
            alfa_m = Mfact/(Rb*bf*h0**2)
        if alfa_m < alfa_r:
            Asc = 0
            As = (Rb*bf*h0*(1-(1-2*alfa_m)**0.5))/Rs
            return (f"Растянутая арматура: {round(As*10**6,2)} мм2 , Сжатая арматура: {round(Asc*10**6,2)} мм2")
        else:
            print('В ребре')
            alfa_m = (Mfact - (Rb*hf*(bf-b)*(h0-0.5*hf)))/(Rb*b*h0**2)
        if alfa_m > alfa_r:
            Asc = (Mfact-alfa_r*Rb*b*h0**2)/(Rs*(h0-asc))
            As = Er*Rb*b*h0/Rs + Asc
        else:
            Asc = 0
            As = (Rb*b*h0*(1-(1-2*alfa_m)**0.5)+Rb*hf*(bf-b))/Rs
        return (f"Растянутая арматура: {round(As*10**6,2)} мм2 , Сжатая арматура: {round(Asc*10**6,2)} мм2")


class Rect(Concrete, Rebars):
    def __init__(self,Con,clss):
        Concrete.__init__(self, Con)
        Rebars.__init__(self, clss)

    def podbor(self,b,h,a,asc,Mfact):
        global Asc, As
        self.h = h
        self.b = b
        self.a = a
        self.asc = asc
        self.Mfact = Mfact
        Rb = Concrete.Rb_n(self)/1.3
        h0 = h-a
        Rs = Rebars.Rs_n(self)/1.15
        Es = Rebars.E_s(self)
        alfa_m = Mfact/(Rb*b*h0**2)
        Er = 0.8/(1+(Rs/Es/0.0035))
        alfa_r = Er*(1-0.5*Er)
        if alfa_m > alfa_r:
            Asc = (Mfact-alfa_r*Rb*b*h0**2)/(Rs*(h0-asc))
            As = Er*Rb*b*h0/Rs + Asc
        else:
            Asc = 0
            As = Rb*b*h0*(1-(1-2*alfa_m)**0.5)/Rs
        return (f"Растянутая арматура: {round(As*10**6,2)} мм2 , Сжатая арматура: {round(Asc*10**6,2)} мм2")


    def proverka(self,b,h,a,asc,Mfact,As,Asc):
        self.h = h
        self.b = b
        self.a = a
        self.asc = asc
        self.Mfact = Mfact
        self.As = As
        self.Asc = Asc
        Rb = Concrete.Rb_n(self)/1.3
        h0 = h-a
        Rs = Rebars.Rs_n(self)/1.15
        Rsc_short = Rebars.Rsc_short(self)
        Es = Rebars.E_s(self)
        Er = 0.8/(1+(Rsc_short/Es/0.0035)) # Кси-эр
        x = (Rs*As-Rsc_short*Asc)/(Rb*b) # сжатая зона бетона, м
        E = x/h0 #кси
        alfa_r = Er*(1-0.5*Er)

        if E < Er:
            Mult = Rb*b*x*(h0-0.5*x) + Rsc_short*Asc*(h0-asc)
            if Mfact < Mult:
                return(f"Прочность сечения обеспечена: {round(Mult,3)} > {Mfact}")
            else:
                return("Прочность не обеспечена")
        else:
            x = Er * h0
            Mult = alfa_r*Rb*b*h0**2* + Rsc_short*Asc*(h0-asc)
            if Mfact < Mult:
                return(f"Прочность сечения обеспечена: {round(Mult,3)} > {Mfact}")
            else:
                alfa_m = E*(1-0.5*E)
                alfa_r = 0.7*alfa_r+0.3*alfa_m
                Mult = alfa_r*Rb*b*h0**2 + Rsc_short*Asc*(h0 - asc)
                if Mfact < Mult:
                    return(f"Прочность сечения обеспечена: {round(Mult,3)} > {Mfact}")
                else:
                   return("Прочность сечения не обеспечена с учетом снижения alfa_m")


    def crack(self,b,h,a,asc,Mfact,Msls_sh,Msls_lg,ds = 16*10**-3, acrc_ult_lg=0.3, acrc_ult_sh=0.4):
        self.h = h
        self.b = b
        self.a = a
        self.asc = asc
        self.Msls_sh = Msls_sh
        self.Msls_lg = Msls_lg
        self.Mfact = Mfact
        h0 = h-a
        eb1_red = 0.0015
        Eb= Concrete.Eb(self)
        Es = Rebars.E_s(self)
        Rb_n= Concrete.Rb_n(self)
        Rbt_n= Concrete.Rbt_n(self)
        Eb_red =  Rb_n/eb1_red
        Rect(self.Con, self.clss).podbor(b,h,a,asc,Mfact)
        alf_s1 = Es/Eb_red
        alfa = Es/Eb
        Ab = h*b
        Ared = Ab + As*alfa + Asc*alfa
        Ired = b*h**3/12 + As*((0.5*h0)**2)*alfa + Asc*((0.5*h0)**2)*alfa
        Sred = b*h*0.5*h + As*a*alfa + Asc*(h-asc)*alfa
        yt = Sred/Ared
        Wred = Ired/yt
        # ex = Wred/Ared
        Wpl = Wred*1.3
        Mcrc = Wpl*Rbt_n
        Mu_s = As/(b*h0)
        Xm = h0*((((Mu_s*alf_s1)**2)+2*Mu_s*alf_s1)**0.5-Mu_s*alf_s1)
        Ired_xm = b*Xm**3/12 + b*Xm*((0.5*Xm)**2)+As*((h0-Xm)**2)*alf_s1+Asc*((Xm-asc)**2)*alf_s1
        Sig_s_sh = Msls_sh*(h0-Xm)*alf_s1/Ired_xm
        Sig_s_lg = Msls_lg*(h0-Xm)*alf_s1/Ired_xm
        Sig_s_crc = Mcrc*(h0-Xm)*alf_s1/Ired_xm
        ksi_s_sh = 1- 0.8*Sig_s_crc/Sig_s_sh
        ksi_s_lg = 1 - 0.8*Sig_s_crc/Sig_s_lg
        Abt = min(max(2*a*b, b*yt),0.5*h*b)
        
        Ls = min(max(10*ds, 0.1,0.5*Abt*ds/As),40*ds,0.4)
        ksi2 = 0.5
        if self.clss == 'A240':
            ksi2 = 0.8
        acrc_1 = 1.4 * ksi2 * 1 *ksi_s_lg* Sig_s_lg * Ls/Es
        acrc_2 = 1 * ksi2 * 1 * ksi_s_sh * Sig_s_sh * Ls/Es
        acrc_3 = 1 * ksi2 * 1 * ksi_s_lg * Sig_s_lg * Ls/Es
        acrc_sh = acrc_1 + acrc_2 - acrc_3
        if acrc_1*10**3 > acrc_ult_lg or acrc_sh*10**3 > acrc_ult_sh:
            return (f"Не проходит по трещиностойкости Acrc_long={round(acrc_1*10**3,2)} > {acrc_ult_lg} мм, \nAcrc_short={round(acrc_sh*10**3,2)} > {acrc_ult_sh}")
        else:
            return (f"Проходит по трещиностойкости Acrc_long={round(acrc_1*10**3,2)} < {acrc_ult_lg} мм,\nAcrc_short={round(acrc_sh*10**3,2)} < {acrc_ult_sh}")


def Prop(Con,Reb):
    # global Rb,Rs,Es,Rsc_long,Rsc_short,Rsn
    d_con = pd.read_csv('Concrete.csv', sep=';', index_col=0)
    Rb = d_con[Con]['Rb_heavy']

    d_reb = pd.read_csv('Reinf_steel.csv', sep=';', index_col=0)
    Rs = d_reb['Rs'][Reb]
    Es = d_reb['Es'][Reb]
    Rsc_long = d_reb['Rsc_long'][Reb]
    Rsc_short = d_reb['Rsc_short'][Reb]
    Rsn = d_reb['Rsn'][Reb]
    return [Rb,Rs,Es,Rsc_long,Rsc_short,Rsn]

# Данные по арматуре площадь(мм2) и количество
rebars = [8,10,12,16,20,25,28,32]
cnt1 = [50,79,113,201,314,491,616,804]
cnt2 = [101,157,226,402,628,982,1232,1608]
cnt3 =[151,236,339,603,942,1473,1847,2413]
cnt4 =[201,314,452,804,1257,1963,2463,3217]
cnt5 =[251,393,565,1005,1571,2454,3079,4021]
cnt6 =[302,471,679,1206,1885,2945,3695,4825]
cnt7 =[352,550,792,14007,2199,3436,4310,5630]
cnt8 =[402,628,905,1608,2513,3927,4926,6434]
cnt9 =[452,707,1018,1810,2827,4418,5542,7238]
cnt10 =[503,785,1131,2011,3142,4909,6158,8042]

data = np.zeros(8, dtype={'names':('rebars','1','2','3','4','5','6','7','8','9','10'),'formats':('i4','i4','i4','i4','i4','i4','i4','i4','i4','i4','i4')})
data['rebars'] = rebars
data['1'] = cnt1
data['2'] = cnt2
data['3'] = cnt3
data['4'] = cnt4
data['5'] = cnt5
data['6'] = cnt6
data['7'] = cnt7
data['8'] = cnt8
data['9'] = cnt9
data['10'] = cnt10

def podbor_arm(L,a,Atreb):
    '''
    L = 200 # ширина балки, мм
    a = 30 # защитный слой, мм
    Atreb = 500 # требуемая площадь армирования, мм
    '''
    for i in range(1,10):
        if (Atreb < data[str(i)]).any():
            diam=np.min(data[data[str(i)]>Atreb]['rebars'])
            l_v_svety = max(diam,25)
            cnt_arm = int((L - a)/(diam+l_v_svety))
            if L-((cnt_arm-1)*(diam+l_v_svety)+a) < a+diam:
                cnt_arm = cnt_arm - 1
            if i <= cnt_arm:
                Sreb = data[np.where(data['rebars']==diam)][0][i]
                return (f"{i} шт Ø{diam}мм А500С, К.И:  {round(Atreb/Sreb,2)}")
            else:
                return ("Ширина балки не достаточна")

Mfact = 0.62 # МН*м
Msls_sh = 0.50 # МН*м
Msls_lg = 0.216 # МН*м
yb1 = 1

h = 0.7
hf = 0.2
b = 0.4
bf = 2.5
a = 0.05
asc = 0.05
# Asc = 4650*10**-6
# As = 5220*10**-6

# prv = Rect("B25","A500").proverka(b,h,a,asc,Mfact,As,Asc)
# print(prv)

print(podbor_arm(300,25,2587))

tty = Rect("B25","A500").crack(b,h,a,asc, Mfact,Msls_sh,Msls_lg,ds = 32*10**-3)
print(tty)
tty = Rect("B25","A500").podbor(b,h,a,asc,Mfact)
print(tty)