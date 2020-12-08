"""Objektorientiertes Programm

Autos werden erstellt und können geändert werden."""

import numpy as np
import matplotlib.pyplot as plt
import simpy
import math
from math import floor, log10
import progressbar
np.random.seed(10)

#%%

class AusEnvironment(simpy.Environment):
    def __init__(self, neutralyr, ausstoss_gesamt=0, anzahl_e_autos=0, gesamt_autos=0):
        self.ausstoss_gesamt = ausstoss_gesamt
        self.anzahl_e_autos = anzahl_e_autos
        self.gesamt_autos = gesamt_autos
        self.neutralyr = neutralyr
        super().__init__()
    def time_and_place(self, car, step, id):
        if id == 'go_elec':
            car.gefahrenekm += car.driving*step
            eausstoss = car.driving*step*car.e_verbrauch*strommix(2020 + self.now*step, self.neutralyr)
            self.ausstoss_gesamt += eausstoss
            #print('(E) {2} hat mal wieder {0}kgCO2 augestoßen. Gesmat CO2-Ausstoß: {1}kg'.format(eausstoss, self.ausstoss_gesamt, car.name))
            return self.timeout(1)
        elif id == 'go_comb':
            car.gefahrenekm += car.driving*step
            self.ausstoss_gesamt += car.driving*step*car.verbrauch
            #print('(C) {2} hat mal wieder {0}kgCO2 augestoßen. Gesmat CO2-Ausstoß: {1}kg'.format(car.driving*step*car.verbrauch, self.ausstoss_gesamt, car.name))
            return self.timeout(1)
        elif id == 'neu_elec':
            #print('verbrauch = {}, e_verbrauch= ={}, driving={}, e_mix={}'.format(car.verbrauch,car.e_verbrauch,car.driving,strommix(2020 + self.now*step, 2050)))
            self.ausstoss_gesamt += car.e_neu_ausstoss
            #print(car.e_neu_ausstoss)
            car.gefahrenekm = 0
            car.type = 'elec'
            self.anzahl_e_autos += 1
            # print('es gibt jetzt {0} E-Autos'.format(self.anzahl_e_autos))
            # print('{0} hat ein neues Elektr-Auto'.format(car.name))
            return self.timeout(0)
        elif id == 'neu_comb':
            self.ausstoss_gesamt += car.neu_ausstoss
            car.gefahrenekm = 0
            car.type = 'comb'
            # print('{0} hat ein neues Verbrenner-Auto'.format(car.name))
            #print(car.neu_ausstoss)
            return self.timeout(0)
        else:
            print('FEHLER: {0} macht nichts'.format(car.name))

#%%

class Car():
    """Car class to implement model functionality
    x: List, keeps track of car position over time
    x_0: Integer, starting position of car
    """
    def __init__(self, step, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, name='Bernd', year=2020, type='comb'):
        self.env = env
        self.lifekm = lifekm                    # Lebensdauer in km
        self.gefahrenekm = gefahrenekm          # beretis gefahrene km
        self.driving = driving                  # gefahrene Kilometer pro Jahr
        self.neu_ausstoss = neu_ausstoss        # kgCO_2-Ausstoß Produktion
        self.e_neu_ausstoss = e_neu_ausstoss    # kgCO_2-Ausstoß Produktion E-Auto
        self.verbrauch = verbrauch              # kg pro km
        self.e_verbrauch = e_verbrauch          # kWh pro km

        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())
        self.name = name
        self.year = year
        self.type = type
        self.env.gesamt_autos += 1
        if type=='elec':
            self.env.anzahl_e_autos+=1
        self.step = step
    def new_e(self):
        self.env.time_and_place(self, self.step, 'neu_elec')

    def run(self):
        while True:
            if self.gefahrenekm < self.lifekm:
                if self.type == 'comb':
                    yield self.env.time_and_place(self, self.step, 'go_comb')
                elif self.type == 'elec':
                    yield self.env.time_and_place(self, self.step, 'go_elec')
                else:
                    print('FEHLER --> welcher Typ ist das Auto?')
            else:
                if self.type == 'comb':
                    a = np.random.random()
                    #if eitheror(a) == 'elec':
                    t = self.env.now*self.step
                    if prod_Kurve1(t, self.env.anzahl_e_autos, self.env.gesamt_autos) == 'elec':
                        yield self.env.time_and_place(self, self.step, 'neu_elec')
                    else:
                    #elif eitheror(a) == 'comb':
                        yield self.env.time_and_place(self, self.step, 'neu_comb')

                #     if eitheror(a) == 'elec':
                #         yield self.env.time_and_place(self, self.step, 'neu_elec')
                #     if eitheror(a) == 'comb':
                #     #elif eitheror(a) == 'comb':
                #         yield self.env.time_and_place(self, self.step, 'neu_comb')
                elif self.type == 'elec':
                    self.env.anzahl_e_autos -= 1
                    yield self.env.time_and_place(self, self.step, 'neu_elec')
                else:
                    print('FEHLER-kein Autotyp festgelegt')

class Car2():
    """Car class to implement model functionality
    x: List, keeps track of car position over time
    x_0: Integer, starting position of car
    ONLY COMBUSTION
    """
    def __init__(self, step, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, name='Bernd', year=2020, type='comb'):
        self.env = env
        self.lifekm = lifekm                    # Lebensdauer in km
        self.gefahrenekm = gefahrenekm          # beretis gefahrene km
        self.driving = driving                  # gefahrene Kilometer pro Jahr
        self.neu_ausstoss = neu_ausstoss        # kgCO_2-Ausstoß Produktion
        self.e_neu_ausstoss = e_neu_ausstoss    # kgCO_2-Ausstoß Produktion E-Auto
        self.verbrauch = verbrauch              # kg pro km
        self.e_verbrauch = e_verbrauch          # kWh pro km

        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())
        self.name = name
        self.year = year
        self.type = type
        self.step = step
        if type=='elec':
            self.env.anzahl_e_autos+=1
        self.env.gesamt_autos += 1


    def run(self):
        while True:
            if self.type=='comb':
                if self.gefahrenekm < self.lifekm:
                    yield self.env.time_and_place(self, self.step, 'go_comb')
                else:
                    yield self.env.time_and_place(self, self.step, 'neu_comb')
            elif self.type=='elec':
                if self.gefahrenekm < self.lifekm:
                    yield self.env.time_and_place(self, self.step, 'go_elec')
                else:
                    yield self.env.time_and_place(self, self.step, 'neu_elec')
            else:
                print('FEHLER in Car2')

#%%

def prod_Kurve(t, anzahl_e_autos, gesamt_autos):
    r = exp                                        # momentan realistischer Anstieg an möglicher Produktion
    fkt = 136617/47700000 * gesamt_autos * np.exp(r*t)
    if anzahl_e_autos < fkt:
        return('elec')
    else:
        return('comb')

def prod_fkt(t, gesamt_autos):
    r = exp                                       # momentan realistischer Anstieg an möglicher Produktion
    fkt = 136617/47700000 * gesamt_autos * np.exp(r*t)
    return(fkt)

def eitheror(a):
    """Funktion die bestimmt, ob neues Auto Elektroauto wird oder nicht.
    In dem Fall X Wahrscheinlichkeit"""
    return('elec')          # X = 100%
    #if a < 0.5:            # X = 50%
    #    return('elec')
    #else:
    #    return('comb')

def aeltesteautosersetzen(autos, anz):
    """Funktion die die ältesten Verbrenner durch Elektrautos ersetzt
    autos: Liste mit allen autos
    anz:   Wie viele Autos ersetzt werden"""
    autos.sort(key=lambda x: (x.type, -x.gefahrenekm), reverse = True)
    for i in np.arange(anz):
        autos[-(i+1)].new_e()

def strommix(currentyr, neutralyr):
    """Strommix über Zeit
    Annahme: linearer Abfall an CO_2 Emissionen pro kWh
    Momentan: 400g pro kWh im Jahr 2020
    neutral         Jahr in dem Dtl 100% Ökostrom hat
    return kg pro kWh zur zeit t
    """
    pa = -0.401/(neutralyr-2019)                     # pro Jahr
    erg = 0.401+pa*(currentyr-2019)
    return max(0, erg)
def parameters(step, env, N, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,year, type):
    sample = np.arange(N)
    for x in sample:
        #print('Auto {0} wird geschaffen'.format(x))
        gefahrenekm = lifekm*np.random.random()
        Car(env, step, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 'auto {0}'.format(x), year, type)


def parametersums(step, env, env2 , N, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,year, type):
    sample = np.arange(N)
    for x in sample:
        #print('Auto {0} wird geschaffen'.format(x))
        #gefahrenekm = lifekm*np.random.random()
        Car(step, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, type)
        Car2(step, env2, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, type)

def parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,year, type):
    sample = np.arange(N)
    Carlist = []
    for x in sample:
        #print('Auto {0} wird geschaffen'.format(x))
        if env.gesamt_autos<((47700000-136617)/47700000*N):
            gefahrenekm = lifekm*np.random.random()
            car = Car(step, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, type)
            Car2(step, env2, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, type)
            Carlist.append(car)
        else:
            gefahrenekm = lifekm*np.random.random()
            Car(step, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, 'elec')
            Car2(step, env2, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss,'auto {0}'.format(x), year, 'elec')
            Carlist.append(car)
    #print('ES GIBT {} EVS'.format(env.anzahl_e_autos))
    return Carlist

def prod_Kurve1(t, anzahl_e_autos, gesamt_autos):
    if gl==1:
        return('elec')
    elif gl==2:
        r = exp                                        # momentan realistischer Anstieg an möglicher Produktion
        fkt = 136617/47700000 * gesamt_autos * np.exp(r*t)
        if anzahl_e_autos < fkt:
            return('elec')
        else:
            return('comb')
    else:
        fkt = anstieglin * gesamt_autos * t
        if anzahl_e_autos < fkt:
            return('elec')
        else:
            return('comb')

def collect(env, T, step):
    ausstoss_data = []
    ausstoss_data.append([0, 0, 0])
    for i in range(int(T/step)):
        env.run(until=i+1)
        ausstoss_data.append([env.now*step, env.ausstoss_gesamt, env.anzahl_e_autos])
    return np.array(ausstoss_data)

def collect_and_fct(env,step, T, autos):
    ausstoss_data = []
    ausstoss_data.append([0, 0, 0])

    for i in range(int(T/step)):
        if env.anzahl_e_autos<env.gesamt_autos:
            t = env.now*step
            mgl = math.floor(prod_fkt(t, env.gesamt_autos)-env.anzahl_e_autos)+1       # berechnet wie viele Autos produziert werden könnten
            ver = env.gesamt_autos -env.anzahl_e_autos  # berechnet wie viele Verbrenner es gibt
            aeltesteautosersetzen(autos, min(mgl, ver))
        env.run(until=i+0.5)
        ausstoss_data.append([env.now*step, env.ausstoss_gesamt, env.anzahl_e_autos])
        #print(env.now/T)
    return np.array(ausstoss_data)

def amor_time(t, array):
    """Amortization time fuer beliebiges Array ausrechnen
    t       Zeitenarray
    Array   Wertearray"""
    for i in np.arange(len(array)):
        if array[i]<0:
            return t[i]
            break
    return(0)


def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

#%%
"""Linearer Umstieg 100%"""
def main ():
    # Parameters
    N = 1
    T = 30
    lifekm = 210000
    gefahrenekm = 210000
    driving = 14000
    neutralyr = 2050
    neu_ausstoss = 9100
    e_neu_ausstoss = 15900
    verbrauch = 0.1274               #kgCO_2/km
    e_verbrauch = 0.18             #kWh/km
    pkt_pto_jahr = 10               # Wie oft soll Gesamtausstoss pro Jahr abgefasst werden
    step = 1/pkt_pto_jahr           # daraus resultierende Schritte
    global exp
    exp = 0.428
    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    which = int(input('1 - Linear\n2 - log1\n3 - log Params\n4 - log rel\n5 - WB\n6 - WB rel\n7 - WB rel vergleich\n8 - WB params\n'))
    global gl
    global anstieglin
    anstieglin = 1
    gl = 0
    bar.start()

    if which ==1:
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 51+b, 10)
        xmax = 50
        fig, ((ax4,ax5,ax6),(ax1,ax2,ax3)) = plt.subplots(2,3, figsize=(15,8))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        """Plot 1,2"""
        gl = 1
        factor = 47.7*10**6/N
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect(env, T, step)
        data2 = collect(env2, T, step)

        G = data1[:,1]-data2[:,1]
        G
        t = data2[:,0]
        ax4.set_title('N = 1')            # Titel fuer Plots
        ax1.set_ylim(-5*round_sig(1.2*np.max(factor*G/1000)), round_sig(1.2*np.max(factor*G/1000)))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Net total $CO_2$ emissions in t')   # Beschriftung y-Achse
        ax1.plot(t+2020, factor*G/1000)
        bar.update(100/3)
        """Plot 1,1"""
        ax4.plot(t+b,factor*data1[:,2])
        ax4.set_ylabel('Amount of EVs')

        """Plot 2,2"""
        N=100
        factor = 47.7*10**6/N

        gl = 0
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect(env, T, step)
        data2 = collect(env2, T, step)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax5.set_title('N = 100')            # Titel fuer Plots
        ax2.set_ylim(-5*round_sig(1.2*np.max(factor*G/1000)), round_sig(1.2*np.max(factor*G/1000)))                             # Achsenlimits y-Achse
        ax2.plot(t+2020, factor*G/1000)
        """Plot 2,1"""
        ax5.plot(t+b,factor*data1[:,2])
        bar.update(2*100/3)

        """Plot 3,2"""
        N=10000
        factor = 47.7*10**6/N
        gl = 0
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect(env, T, step)
        data2 = collect(env2, T, step)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax6.set_title('N = 10000')            # Titel fuer Plots
        ax3.set_ylim(-5*round_sig(1.2*np.max(factor*G/1000)), round_sig(1.2*np.max(factor*G/1000)))                             # Achsenlimits y-Achse
        ax3.plot(t+2020, factor*G/1000)
        print('The amortization time is {0}'.format(amor_time(t, G)))
        """Plot 3,1"""
        ax6.plot(t+b,factor*data1[:,2])
        #fct = anstieglin * N * t
        #fct[anstieglin * N * t > N] = N
        #ax6.plot(t+b, fct)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        xlabel = r'Time t in Years'
        ax1.set_xlabel(xlabel)                # Beschriftung x-Achse
        ax2.set_xlabel(xlabel)                # Beschriftung x-Achse
        ax3.set_xlabel(xlabel)                # Beschriftung x-Achse
        ax4.set_xlabel(xlabel)                # Beschriftung x-Achse
        ax5.set_xlabel(xlabel)                # Beschriftung x-Achse
        ax6.set_xlabel(xlabel)                # Beschriftung x-Achse
        bar.update(100)
        ax1.axhline(y=0, c='k', ls='--')
        ax2.axhline(y=0, c='k', ls='--')
        ax3.axhline(y=0, c='k', ls='--')
        ax1.annotate(r"$\Delta t_{a} = 5.5$ years", xy=(2025.5, 0), xytext=(2030, 0.15*10**9),arrowprops=dict(arrowstyle="->"), size = 10)
        ax3.annotate(r"$\Delta t_{a} = 9.9$ years", xy=(2029.92, 0), xytext=(2035, -1*10**8),arrowprops=dict(arrowstyle="->"), size = 10)
        ax1.text(0.05, 0.95, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.05, 0.95, 'e)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.05, 0.95, 'f)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax4.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax5.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
        ax6.text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/21lin.pdf")


    elif which ==2:
        T=30
        gl=2
        N=10000
        b=2020
        factor = 47.7*10**6/N
        fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(15,5))
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect(env, T, step)
        data2 = collect(env2, T, step)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax1.set_title('Net total emissions $G_{rsc}(t)$')            # Titel fuer Plots
        ax1.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(-1.2*np.max(G/1000*factor), 1.2*np.max(G/1000*factor))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Net total $CO_2$ emissions in t')   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in Years')                # Beschriftung x-Achse
        ax1.plot(t+b, G/1000*factor)
        ax1.axhline(y=0, color='k',ls='--')
        #ax1.plot(t,(data1[:,1]-data2[:,1])/data2[:,1])
        print('The amortization time is {0}'.format(amor_time(t, G)))
        ges_e = data1[:,2]
        t = data2[:,0]
        ax2.plot(t+b, ges_e*factor)
        ax2.plot(t+b, 136617/47700000 * N * np.exp(0.428*t)*factor, linestyle = ':', color = 'r')
        ax2.set_title('Growth function $N^{tot}_{rsc}(t)$')            # Titel fuer Plots
        ax2.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(-0.2*np.max(ges_e*factor), 1.2*np.max(ges_e*factor))                             # Achsenlimits y-Achse
        ax2.set_ylabel(r'Total amount of EVs')   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in Years')                # Beschriftung x-Achse

        grad=np.gradient(ges_e*factor/step)
        np.shape(grad)
        con=10
        vals = np.convolve(grad,np.ones(con),mode='valid')
        vals
        ax3.plot(t[con-1+6:]-0.5+b, vals[6:]/con)
        ax3.plot(t+b, 136617/47700000 *0.428* N * np.exp(0.428*t)*factor, linestyle = ':', color = 'r')
        ax3.set_title(r'Growth rate $\nu_{rsc}(t)$')            # Titel fuer Plots
        ax3.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(-0.2*np.max(grad), 1.2*np.max(grad))                             # Achsenlimits y-Achse
        ax3.set_ylabel(r'Growth rate in $\frac{Vehicles}{Year}$')   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in Years')                # Beschriftung x-Achse
        ax1.annotate(r"$\Delta t_{a} = 14,8$ years", xy=(2034.79, 0), xytext=(2036, 1*10**7),arrowprops=dict(arrowstyle="->"), size = 10)

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/21rsc.pdf")


    elif which ==3:
        gl=2
        swipe = []
        fig, ((ax1, ax2, ax3)) = plt.subplots(1,3,figsize = (15,6))
        plt.subplots_adjust(wspace=0.3)

        N=10000
        b=2020
        M = 1 #60
        for i in np.arange(M):
            e_neu_ausstoss = neu_ausstoss+i*500/3
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                        neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect(env, T, step)
            data2 = collect(env2, T, step)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            swipe.append([e_neu_ausstoss-neu_ausstoss, amor_time(t,G)])
            bar.update(i/M*100)
        swipe = np.array(swipe)
        ax1.set_title('Net initial emission $a_\mathrm{i}$')
        ax1.set_xlim(swipe[:,0][0],swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax1.set_ylim(0+b, b+20)                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Year of systemic $CO_2$ amortization')   # Beschriftung y-Achse
        ax1.set_xlabel(r'$a_\mathrm{i}$ in t')                # Beschriftung x-Achse
        ax1.plot(swipe[:,0],b+swipe[:,1])
        bar.finish()
        print('#####################Fertig1###################')



        neu_ausstoss = 9100
        e_neu_ausstoss = 14700
        swipe =[]
        M = 1  #50
        bar.start()

        for i in np.arange(M):
            neutralyr = 2021+i
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                        neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect(env, T, step)
            data2 = collect(env2, T, step)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            swipe.append([neutralyr, amor_time(t,G)])
            bar.update(i/M*100)
        swipe = np.array(swipe)
        ax2.set_title('Year of climate neutral power generation $t_\mathrm{CN}$')
        ax2.set_xlim(2020,swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax2.set_ylim(b+0, b+20)                             # Achsenlimits y-Achse
        ax2.set_ylabel(r'Year of systemic $CO_2$-amortization')   # Beschriftung y-Achse
        ax2.set_xlabel(r'$t_\mathrm{CN}$ in years')                # Beschriftung x-Achse
        ax2.plot(swipe[:,0],b+swipe[:,1])

        bar.finish()
        print('#####################Fertig2###################')

        neutralyr = 2050
        swipe =[]
        M = 1 #100
        bar.start()
        for i in np.arange(M):
            exp = i/M
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                        neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect(env, T, step)
            data2 = collect(env2, T, step)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            amor_time(t,G)
            swipe.append([exp, amor_time(t,G)])
            bar.update(i/M*100)
            #plt.plot(t,G)
        swipe = np.array(swipe)
        bar.finish()
        ax3.set_title('Exponential growth factor $c_\mathrm{exp}^\mathrm{rsc}$')
        #ax3.set_xlim(swipe[:,0][0],swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax3.set_ylim(0+b, b+20)                             # Achsenlimits y-Achse
        ax3.set_ylabel(r'Year of systemic $CO_2$-amortization')   # Beschriftung y-Achse
        ax3.set_xlabel(r'$c_\mathrm{exp}^\mathrm{rsc}$')                # Beschriftung x-Achse
        ax3.plot(swipe[:,0],b+swipe[:,1])
        ax1.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        yticks = np.arange(2020,2041,5)
        ax1.set_yticks(yticks)
        ax2.set_yticks(yticks)
        ax3.set_yticks(yticks)

        print('#####################Fertig3###################')
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/23rscparams.pdf")


    elif which ==4:
        T=30
        gl=2
        N=10000
        b = 2020
        factor = 47.7*10**6/N
        fig, (ax1) = plt.subplots(1,1,figsize=(15,7))
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect(env, T, step)
        data2 = collect(env2, T, step)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax1.set_title('relative Emissions', size=15)            # Titel fuer Plots
        ax1.set_xlim(t[0]+b, t[-1]+b)
        ax1.set_ylim(-0.7,0.1)                           # Achsenlimits x-Achse
        # ax1.set_ylim(-1.2*np.max(G/data2[:,1]/1000*factor), 1.2*np.max(G/data2[:,1]/1000*factor))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'relative $CO_2$-emission rate',size=15)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in Years',size=15)                # Beschriftung x-Achse
        #ax1.plot(t, G/1000*factor)
        ax1.plot(t+b,np.gradient(data1[:,1]-data2[:,1])/np.gradient(data2[:,1]))
        print('The amortization time is {0}'.format(amor_time(t, G)))
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/22relem.pdf")


    elif which ==5:
        gl=2
        T=40
        b=2020
        N=10000
        factor = 47.7*10**6/N
        np.random.seed(11)

        fig, (ax2,ax1,ax3) = plt.subplots(1,3, figsize =(15,6))
        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect_and_fct(env, step, T, autos)
        data2 = collect_and_fct(env2, step, T, autos)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax2.set_title('Net total emissions $G_\mathrm{wbs}(t)$')            # Titel fuer Plots
        ax2.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(-5*np.max(G/1000*factor), 1.2*np.max(G/1000*factor))
        #ax.set_ylim(-200000, 100000)                     # Achsenlimits y-Achse
        ax2.set_ylabel(r'Net total $CO_2$ emissions in t')   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years')                # Beschriftung x-Achse
        ax2.plot(t+b, G*factor/1000)
        print('The amortization time is {0}'.format(amor_time(t, G)))
        ax2.axhline(y=0, c='k', ls='--')
        ax2.annotate(r"$\Delta t_\mathrm{a} = 17.8$ years", xy=(2037.83, 0), xytext=(2040, 0.25*10**9),arrowprops=dict(arrowstyle="->"), size = 10)

        ges_e = data1[:,2]
        ges_e
        t = data2[:,0]
        ax1.plot(t+b, ges_e*factor)
        ax1.set_title('Growth function $N^\mathrm{tot}_\mathrm{wbs}(t)$')            # Titel fuer Plots
        ax1.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(-0.2*np.max(ges_e*factor), 1.2*np.max(ges_e*factor))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Total amount of EVs')   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years')

        grad=np.gradient(ges_e*factor/step)

        ax3.plot(t+b,grad)
        ax3.plot(t+b, 136617/47700000 *0.428* N * np.exp(0.428*t)*factor, linestyle = ':', color = 'r')
        ax3.set_title(r'Growth rate $\nu_{wbs}(t)$')            # Titel fuer Plots
        ax3.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(-0.2*np.max(grad), 1.2*np.max(grad))                             # Achsenlimits y-Achse
        ax3.set_ylabel(r'Growth rate in $\frac{Vehicles}{Year}$')   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in Years')
        ax1.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/24wbs0.pdf")



    elif which ==6:
        T=30
        gl=2
        b = 2020
        N=10000
        factor = 47.7*10**6/N
        fig, (ax1) = plt.subplots(1,1)

        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
            neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect_and_fct(env, step, T, autos)
        data2 = collect_and_fct(env2, step, T, autos)
        G = data1[:,1]-data2[:,1]
        t = data2[:,0]
        ax1.set_title('relative Emissions')            # Titel fuer Plots
        ax1.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        # ax1.set_ylim(-1.2*np.max(G/data2[:,1]/1000*factor), 1.2*np.max(G/data2[:,1]/1000*factor))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'relative $CO_2$-emission')   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in Years')                # Beschriftung x-Achse
        #ax1.plot(t, G/1000*factor)
        ax1.plot(t+b,(data1[:,1]-data2[:,1])/data2[:,1])
        print('The amortization time is {0}'.format(amor_time(t, G)))

    elif which==7:
        T=45
        gl=2
        b = 2020
        N=1000
        factor = 47.7*10**6/N
        fig, (ax1) = plt.subplots(1,1)

        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
            neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data1 = collect_and_fct(env, step, T, autos)
        data2 = collect_and_fct(env2, step, T, autos)

        env = AusEnvironment(neutralyr)
        env2 = AusEnvironment(neutralyr)
        parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                    neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
        data12 = collect(env, T, step)
        data22 = collect(env2, T, step)


        G1 = data1[:,1]-data2[:,1]
        G2 = data12[:,1]-data22[:,1]
        G=G1-G2
        t = data2[:,0]
        ax1.set_title('Difference of emission')            # Titel fuer Plots
        ax1.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
        # ax1.set_ylim(-1.2*np.max(G/data2[:,1]/1000*factor), 1.2*np.max(G/data2[:,1]/1000*factor))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Difference of emission $CO_2$-emission')   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in Years')                # Beschriftung x-Achse
        #ax1.plot(t, G/1000*factor)
        ax1.plot(t+b,G/1000*factor)
        print('The amortization time is {0}'.format(amor_time(t, G)))
    elif which ==8:
        gl=2
        swipe = []
        fig, ((ax1, ax2, ax3)) = plt.subplots(1,3)
        N=10000
        b=2020
        M = 60
        for i in np.arange(M):
            e_neu_ausstoss = neu_ausstoss+i*500/3
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect_and_fct(env, step, T, autos)
            data2 = collect_and_fct(env2, step, T, autos)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            swipe.append([e_neu_ausstoss-neu_ausstoss, amor_time(t,G)])
            bar.update(i/M*100)
        swipe = np.array(swipe)
        ax1.set_title('Initial emission')
        ax1.set_xlim(swipe[:,0][0],swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax1.set_ylim(0+b, b+np.max(swipe[:,1]))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Year of systemic $CO_2$-amortization')   # Beschriftung y-Achse
        ax1.set_xlabel(r'Difference of initial emission')                # Beschriftung x-Achse
        ax1.plot(swipe[:,0],b+swipe[:,1])
        bar.finish()
        print('#####################Fertig1###################')



        neu_ausstoss = 9100
        e_neu_ausstoss = 14700
        swipe =[]
        M = 50
        bar.start()

        for i in np.arange(M):
            neutralyr = 2021+i
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect_and_fct(env, step, T, autos)
            data2 = collect_and_fct(env2, step, T, autos)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            swipe.append([neutralyr, amor_time(t,G)])
            bar.update(i/M*100)
        swipe = np.array(swipe)
        ax2.set_title('Climate Neutrality')
        ax2.set_xlim(swipe[:,0][0],swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax2.set_ylim(b+0, b+np.max(swipe[:,1]))                             # Achsenlimits y-Achse
        ax2.set_ylabel(r'Year of systemic $CO_2$-amortization')   # Beschriftung y-Achse
        ax2.set_xlabel(r'Year of climate neutrality')                # Beschriftung x-Achse
        ax2.plot(swipe[:,0],b+swipe[:,1])

        bar.finish()
        print('#####################Fertig2###################')

        neutralyr = 2050
        swipe =[]
        M = 50
        bar.start()
        for i in np.arange(M):
            exp = i/M
            env = AusEnvironment(neutralyr)
            env2 = AusEnvironment(neutralyr)
            autos = parametersums2(step, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
            data1 = collect_and_fct(env, step, T, autos)
            data2 = collect_and_fct(env2, step, T, autos)
            G = data1[:,1]-data2[:,1]
            t = data2[:,0]
            amor_time(t,G)
            swipe.append([exp, amor_time(t,G)])
            bar.update(i/M*100)
            #plt.plot(t,G)
        swipe = np.array(swipe)
        bar.finish()
        ax3.set_title('Exponent r')
        #ax3.set_xlim(swipe[:,0][0],swipe[:,0][-1])                            # Achsenlimits x-Achse
        ax3.set_ylim(0+b, b+np.max(swipe[:,1]))                             # Achsenlimits y-Achse
        ax3.set_ylabel(r'Year of systemic $CO_2$-amortization')   # Beschriftung y-Achse
        ax3.set_xlabel(r'Exponent')                # Beschriftung x-Achse
        ax3.plot(swipe[:,0],b+swipe[:,1])


        print('#####################Fertig3###################')


    bar.finish()
    plt.show()
    plt.draw()

if __name__ == "__main__":
    main()

#%%
