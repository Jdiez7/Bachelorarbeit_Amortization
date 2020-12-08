"""Objektorientiertes Programm

Autos werden erstellt und können geändert werden."""

import numpy as np
import matplotlib.pyplot as plt
import simpy
import math
#np.random.seed(1)

#%%

class AusEnvironment(simpy.Environment):
    def __init__(self, ausstoss_gesamt=0, anzahl_e_autos=0, gesamt_autos=0):
        self.ausstoss_gesamt = ausstoss_gesamt
        self.anzahl_e_autos = anzahl_e_autos
        self.gesamt_autos = gesamt_autos
        super().__init__()
    def time_and_place(self, car, step, id):
        if id == 'go_elec':
            car.gefahrenekm += car.driving*step
            eausstoss = car.driving*step*car.e_verbrauch*strommix(2020 + self.now*step, 2050)
            self.ausstoss_gesamt += eausstoss
            #print('(E) {2} hat mal wieder {0}kgCO2 augestoßen. Gesmat CO2-Ausstoß: {1}kg'.format(eausstoss, self.ausstoss_gesamt, car.name))
            return self.timeout(1)
        elif id == 'go_comb':
            car.gefahrenekm += car.driving*step
            self.ausstoss_gesamt += car.driving*step*car.verbrauch
            #print('(C) {2} hat mal wieder {0}kgCO2 augestoßen. Gesmat CO2-Ausstoß: {1}kg'.format(car.driving*step*car.verbrauch, self.ausstoss_gesamt, car.name))
            return self.timeout(1)
        elif id == 'neu_elec':
            self.ausstoss_gesamt += car.e_neu_ausstoss
            car.gefahrenekm = 0
            car.type = 'elec'
            self.anzahl_e_autos += 1
            #print('es gibt jetzt {0} E-Autos {1}'.format(self.anzahl_e_autos, self.now))
            #print('{0} hat ein neues Elektr-Auto'.format(car.name))
            return self.timeout(0)
        elif id == 'neu_comb':
            self.ausstoss_gesamt += car.neu_ausstoss
            car.gefahrenekm = 0
            car.type = 'comb'
            #print('{0} hat ein neues Verbrenner-Auto'.format(car.name))
            return self.timeout(0)
        else:
            print('FEHLER: {0} macht nichts'.format(car.name))

#%%

class Car():
    """Car class to implement model functionality
    x: List, keeps track of car position over time
    x_0: Integer, starting position of car
    """
    def __init__(self, step, val, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, name='Bernd', year=2020, type='comb'):
        self.env = env
        self.lifekm = lifekm                    # Lebensdauer in km
        self.gefahrenekm = gefahrenekm          # beretis gefahrene km
        self.driving = driving                  # gefahrene Kilometer pro Jahr
        self.neu_ausstoss = neu_ausstoss        # kgCO_2-Ausstoß Produktion
        self.e_neu_ausstoss = e_neu_ausstoss    # kgCO_2-Ausstoß Produktion E-Auto
        self.verbrauch = verbrauch              # kg pro km
        self.e_verbrauch = e_verbrauch          # kWh pro km
        self.val = val

        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())
        self.name = name
        self.year = year
        self.type = type
        self.env.gesamt_autos += 1
        #print('jetzt gibte es insg. {0} Autos'.format(self.env.gesamt_autos))
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
                if self.val == 'scenario1':
                    if self.type == 'comb':

                        t = self.env.now*self.step
                        if prod_Kurve1(t, self.env.anzahl_e_autos, self.env.gesamt_autos) == 'elec':
                            yield self.env.time_and_place(self, self.step, 'neu_elec')
                        else:
                        #elif eitheror(a) == 'comb':
                            yield self.env.time_and_place(self, self.step, 'neu_comb')
                    elif self.type == 'elec':
                        self.env.anzahl_e_autos -= 1
                        yield self.env.time_and_place(self, self.step, 'neu_elec')

                elif self.val == 'scenario2':
                    if self.type == 'comb':
                        #print('combkaputt')
                        a = np.random.random()
                        #if eitheror(a) == 'elec':
                        t = self.env.now*self.step
                        if prod_Kurve2(t, self.env.anzahl_e_autos, self.env.gesamt_autos) == 'elec':
                            yield self.env.time_and_place(self, self.step, 'neu_elec')
                        else:
                        #elif eitheror(a) == 'comb':
                            yield self.env.time_and_place(self, self.step, 'neu_comb')
                    elif self.type == 'elec':
                        #print('EAuto kaputt')
                        self.env.anzahl_e_autos -= 1
                        yield self.env.time_and_place(self, self.step, 'neu_elec')

                elif self.val == 'scenario3':
                    if self.type == 'comb':
                        #print('combkaputt')
                        a = np.random.random()
                        #if eitheror(a) == 'elec':
                        t = self.env.now*self.step
                        if prod_Kurve(t, self.env.anzahl_e_autos, self.env.gesamt_autos) == 'elec':
                            yield self.env.time_and_place(self, self.step, 'neu_elec')
                        else:
                        #elif eitheror(a) == 'comb':
                            yield self.env.time_and_place(self, self.step, 'neu_comb')
                    elif self.type == 'elec':
                        #print('EAuto kaputt')
                        self.env.anzahl_e_autos -= 1
                        yield self.env.time_and_place(self, self.step, 'neu_elec')
                else:
                    print('FEHLER')

class Car2():
    """Car class to implement model functionality
    x: List, keeps track of car position over time
    x_0: Integer, starting position of car
    ONLY COMBUSTION
    """
    def __init__(self, step, val, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, name='Bernd', year=2020, type='comb',):
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
        self.val = val
        self.env.gesamt_autos += 1


    def run(self):
        while True:
            if self.gefahrenekm < self.lifekm:
                yield self.env.time_and_place(self, self.step, 'go_comb')
            else:
                yield self.env.time_and_place(self, self.step, 'neu_comb')

#%%
def prod_Kurve2(t, anzahl_e_autos, gesamt_autos):
    r = 0.44                                        # momentan realistischer Anstieg an möglicher Produktion
    fkt = 34000/50000000 * gesamt_autos * np.exp(r*t)
    if anzahl_e_autos < fkt:
        return('elec')
    else:
        return('comb')


def prod_Kurve1(t, anzahl_e_autos, gesamt_autos):

    fkt = 0.05 * gesamt_autos * t
    if anzahl_e_autos < fkt:
        return('elec')
    else:
        return('comb')

def prod_fkt(t, gesamt_autos):
    r = 0.44                                        # momentan realistischer Anstieg an möglicher Produktion
    fkt = 34000/50000000 * gesamt_autos * np.exp(r*t)
    return(fkt)

def prod_Kurve(t, anzahl_e_autos, gesamt_autos):                                      # momentan realistischer Anstieg an möglicher Produktion
    fkt = prod_fkt(t, gesamt_autos)
    if anzahl_e_autos < fkt:
        return('elec')
    else:
        return('comb')

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
    pa = -0.4/(neutralyr-2020)                     # pro Jahr
    erg = 0.4+pa*(currentyr-neutralyr)
    return max(0, erg)

def parameters(env, N, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, year, type):
    sample = np.arange(N)
    for x in sample:
        #print('Auto {0} wird geschaffen'.format(x))
        gefahrenekm = lifekm*np.random.random()




def parametersums(step, val, env, env2, N, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, year, type):
    sample = np.arange(N)
    Carlist = []
    for x in sample:
        #print('Auto {0} wird geschaffen'.format(x))
        gefahrenekm = lifekm*np.random.random()
        car = Car(step, val, env, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 'auto {0}'.format(x), year, type)
        car2 = Car2(step, val, env2, lifekm, gefahrenekm, driving, neu_ausstoss, verbrauch,e_verbrauch, e_neu_ausstoss, 'auto {0}'.format(x), year, type)
        Carlist.append(car)
    return Carlist

def collect_and_fct(env,step, T, autos):
    ausstoss_data = []
    for i in range(int(T/step)):
        if env.anzahl_e_autos<env.gesamt_autos:
            t = env.now*step
            mgl = math.floor(prod_fkt(t, env.gesamt_autos)-env.anzahl_e_autos)       # berechnet wie viele Autos produziert werden könnten
            ver = env.gesamt_autos -env.anzahl_e_autos  # berechnet wie viele Verbrenner es gibt
            aeltesteautosersetzen(autos, min(mgl, ver))
        env.run(until=i+0.5)
        ausstoss_data.append([env.now*step, env.ausstoss_gesamt, env.anzahl_e_autos])
        #print(env.now/T)
    return np.array(ausstoss_data)

def collect_and_fctohne(env, step, T):
    ausstoss_data = []
    for i in range(int(T/step)):
        env.run(until=i+0.5)
        ausstoss_data.append([env.now*step, env.ausstoss_gesamt, env.anzahl_e_autos])
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

#%%
"""Exponentieller Umstieg"""
def main ():
    # Parameters
    T = 30
    b = 2020
    lifekm = 200000
    gefahrenekm = 200000
    driving = 20000
    neu_ausstoss = 6000
    e_neu_ausstoss = 11000
    verbrauch = 0.2
    e_verbrauch = 0.15
    pkt_pto_jahr = 10                  # Wie oft soll Gesamtausstoss pro Jahr abgefasst werden
    step = 1/pkt_pto_jahr               # daraus resultierende Schritte


    # Erstellen der Plotumgebung
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize= (12,7))
    plt.subplots_adjust(hspace= 0.4, wspace=0.15)

    """FIGURE 1"""
    N = 10
    factor = 50*10**6/N
    val = 'scenario1'
    env = AusEnvironment()
    env2 = AusEnvironment()
    autos = parametersums(step, val, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
    data1 = collect_and_fctohne(env, step, T)
    data2 = collect_and_fctohne(env2, step, T)
    np.shape(data1)

    titlesize = 15
    supsize = 13

    ges_e = data1[:,2]
    t = data2[:,0]
    ax1.plot(t+b, factor*ges_e)
    ax1.set_title('Limited linear growth \nN = {0}'.format(N),size=titlesize)            # Titel fuer Plots
    ax1.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
    ax1.set_ylim(-0.2*factor*np.max(ges_e), 1.2*factor*np.max(ges_e))                             # Achsenlimits y-Achse
    ax1.set_ylabel(r'Total number of EVs $N^{tot}(t)$',size=supsize)   # Beschriftung y-Achse
    ax1.set_xlabel(r'Time $t$ in years',size=supsize)                # Beschriftung x-Achse

    """FIGURE 2"""
    N = 10000
    factor = 50*10**6/N
    val = 'scenario1'
    env = AusEnvironment()
    env2 = AusEnvironment()
    autos = parametersums(step, val, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
    data1 = collect_and_fctohne(env, step, T)
    data2 = collect_and_fctohne(env2, step, T)
    ges_e = data1[:,2]
    t = data2[:,0]
    ax2.plot(t+b, factor*ges_e)
    ax2.set_title('Limited linear growth \nN = {0}'.format(N),size=titlesize)            # Titel fuer Plots
    ax2.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
    ax2.set_ylim(-0.2*factor*np.max(ges_e), 1.2*factor*np.max(ges_e))                             # Achsenlimits y-Achse
    ax2.set_ylabel(r'Total number of EVs $N^{tot}(t)$',size=supsize)   # Beschriftung y-Achse
    ax2.set_xlabel(r'Time $t$ in years',size=supsize)                # Beschriftung x-Achse
    print('erstes fertig')

    """FIGURE 3"""
    N = 10000
    factor = 50*10**6/N
    val = 'scenario2'
    env = AusEnvironment()
    env2 = AusEnvironment()
    autos = parametersums(step,val, env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
    data1 = collect_and_fctohne(env, step, T)
    data1
    data2 = collect_and_fctohne(env2, step, T)
    data2
    ges_e = data1[:,2]
    t = data2[:,0]
    ax3.plot(t+b, factor*ges_e)
    ref = factor*prod_fkt(t,N)
    test = ref<N*factor
    ref1 = ref[test]
    ax3.hlines(N*factor,0+b, 30+b, 'k', ls='--', lw=1)
    ax3.plot(t[test]+b, ref1, 'r--')
    ax3.set_title('Realistic Scenario \nN = {0}'.format(N),size=titlesize)            # Titel fuer Plots
    ax3.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
    ax3.set_ylim(-0.2*np.max(ges_e)*factor, 1.2*np.max(ges_e)*factor)                             # Achsenlimits y-Achse
    ax3.set_ylabel(r'Total number of EVs $N^{tot}(t)$',size=supsize)   # Beschriftung y-Achse
    ax3.set_xlabel(r'Time $t$ in years',size=supsize)                # Beschriftung x-Achse
    print('zweites fertig')

    """FIGURE 4"""
    N = 10000
    factor = 50*10**6/N
    val = 'scenario3'
    env = AusEnvironment()
    env2 = AusEnvironment()
    autos = parametersums(step,val,  env, env2, N, lifekm, gefahrenekm, driving,
                neu_ausstoss, verbrauch, e_verbrauch, e_neu_ausstoss, 2020, 'comb')
    data1 = collect_and_fct(env, step, T, autos)
    data2 = collect_and_fct(env2, step, T, autos)
    ges_e = data1[:,2]
    t = data2[:,0]
    ax4.plot(t+b, factor*ges_e)
    ax4.set_title('Wrecking Bonus Scenario \nN = {0}'.format(N),size=titlesize)            # Titel fuer Plots
    ax4.set_xlim(t[0]+b, t[-1]+b)                            # Achsenlimits x-Achse
    ax4.set_ylim(-0.2*factor*np.max(ges_e), 1.2*factor*np.max(ges_e))                             # Achsenlimits y-Achse
    ax4.set_ylabel(r'Total number of EVs $N^{tot}(t)$',size=supsize)   # Beschriftung y-Achse
    ax4.set_xlabel(r'Time $t$ in years',size=supsize)                # Beschriftung x-Achse
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax1.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
    ax2.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
    ax3.text(0.05, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =13)
    ax4.text(0.05, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =13)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/011difgrowth.pdf")
    plt.show()
    plt.draw()


if __name__ == "__main__":
    main()

 #%%
