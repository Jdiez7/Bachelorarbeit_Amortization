import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib

def single_simple_delta(t, argsb ):
    """Emissionsfunktion eines einzelnen Units:
    t           Zeitarray
    a           Anfangsemission
    b           Einsparung pro  Jahr
    lifespan    Lebensdauer
    delta       Breite delta-fkt"""
    a = args['a']
    b = args['b']
    lifespan = args['lifespan']
    delta = args['delta']
    dirac = np.exp(-t**2/delta**2)/(np.sqrt(np.pi)*delta)
    return  dirac + b*np.heaviside(t,0)*np.heaviside(- t + lifespan, 0)

def single_simple_block(t, args):
    """Emissionsfunktion eines einzelnen Units:
    t           Zeitarray
    a           Anfangsemission
    b           Einsparung pro  Jahr
    lifespan    Lebensdauer"""
    a = args['a']
    b = args['b']
    lifespan = args['lifespan']
    breite = args['breite']
    dx = (t[-1]-t[0])/(len(t)-1)
    y = b*np.heaviside(t,0)*np.heaviside(- t + lifespan, 0)
    for i in np.arange(len(t)):
        if t[i]>0:
            for m in np.arange(int(breite)):
                y[i+m] += a/(breite*dx)
            break
    return y

def strommix(t, args):
    """Strommix über Zeit
    Annahme: linearer Abfall an CO_2 Emissionen pro kWh
    Momentan: 400g pro kWh im Jahr 2020
    neutral         Jahr in dem Dtl 100% Ökostrom hat
    return kg pro kWh zur zeit t
    """
    neutral = args['neutral']
    pa = -0.4/(neutral-2020)                     # pro Jahr
    erg = 0.4+pa*t
    return erg.clip(min=0)


def single_simple_real(t, args):
    """
    alles wird in kg gerechnet
    a                           Anfangsemission
    b                           Ausstoss Verbrenner pro Jahr
    """
    lifekm, driving, breite, neu_ausstoß, neu_ausstoß_ICE, verbrauch, recycling , ausstoss_ICE = args['lifekm'], args['driving'], args['breite'], args['neu_ausstoß'], args['neu_ausstoß_ICE'], args['verbrauch'], args['recycling'], args['ausstoss_ICE']
    lifetime = lifekm/driving
    a = neu_ausstoß
    dx = (t[-1]-t[0])/(len(t)-1)
    einsparung = driving*verbrauch*strommix(t, args)-ausstoss_ICE*driving


    # Einsparung pro jahr
    y = einsparung*np.heaviside(t,0)*np.heaviside(- t + lifetime, 0)   #- breite*dx


    i = np.searchsorted(t,0,side='left')
    y[i] += a/dx - breite*a
    for m in np.arange(int(breite)):
        y[i-m] += a

    j = np.searchsorted(t, lifetime+breite*dx, side='left')
    y[j] += recycling/dx - breite*recycling
    for m in np.arange(int(breite)):
        y[j+m] += recycling


    return y


"""ab hier könnte auch alles importiert werden, weiß gerade nur nicht wie das geht"""
def faltung(v, f, t):
    """Faltung der Net emission of a single unit mit growth fkt
    f       net emission of a single Unit
    v       growth funktion
    """
    dx = (t[-1]-t[0])/(len(t)-1)
    g = np.zeros(len(t))
    # herausfinden wo t = 0
    for i in np.arange(len(t)):
        if t[i] >= 0:
            t_0 = i
            break
    # Faltung ab t = 0
    for i in np.arange(len(t)-t_0):
        g[t_0+i]= np.sum(v[t_0:t_0+i+1]*np.flip(f[t_0:t_0+i+1])*dx)
    return g

def growth_exp(t, args):
    """Wachstumsfunktion Zubau; hier Annahme exponentiell
    t           Zeitenarray
    r           Exp. Rate"""
    r, a_0 = args['r'], args['a_0']
    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        if t[i]>0:
            a[i:] = a_0*np.exp(r*t[i:])
            break
    return a

def growth_lin(t, args):
    """Wachstumsfunktion Zubau; hier Annahme linear
    t           Zeitenarray
    c           Zubaurate"""
    c = args['c']
    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        if t[i]>0:
            a[i:] = c
            break
    return a

# def growth_log_int(t, args):
#     """Wachstumsfunktion Zubau; hier Annahme logistisch
#     t           Zeitenarray
#     r           Exp. Rate"""
#     k, amax = args['k'], args['amax']
#     return amax/(1+np.exp(-k*(t-xhalb)))

    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        if t[i]>0:
            a[i:] = a_0*amax/(a_0+(amax-a_0)*np.exp(-amax*k*t[i:]))
            break
    return a

def growth_log(t, args):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    k, amax, a_0 = args['k'], args['amax'], args['a_0']


    a = np.zeros(len(t))
    a = amax*(amax/a_0-1)*amax*k*np.exp(-amax*k*t)/(((amax/a_0-1)*np.exp(-amax*k*t)+1))**2
    return a

def amor_time(t, array):
    """Amortization time fuer beliebiges Array ausrechnen
    t       Zeitenarray
    Array   Wertearray"""
    for i in np.arange(len(array)):
        if array[i]<0:
            return t[i]
            break
    return(0)

def func(label):
    """RadioButton"""
    hzdict = {'1': 1, '2': 2, '3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'10': 10}
    global which
    which = hzdict[label]

def main():
    args = {}
    args['a'] = 2                       # Anfangsemission
    args['b'] = -1                      # Ausstoss Verbrenner pro Jahr
    args['lifespan'] = 10               # Lebensdauer
    args['delta'] = 0.1                 # breite der Deltafunktion
    args['n'] = 10000

    # REALISTISCH
    args['lifekm'] = 200000             # Lebensdauer in km
    args['driving'] = 20000             # gefahrene Kilometer pro Jahr
    args['breite'] = 0    #args['n']/500
    args['neutral'] = 2035
    args['neu_ausstoß'] = 11000          # kgCO_2-Ausstoß EV
    args['neu_ausstoß_ICE'] = 6000      # kgCO_2-Ausstoß ICE
    args['verbrauch'] = 0.15            # kWh pro km
    args['ausstoss_ICE'] = 0.2          # kg pro km
    args['recycling'] = 0             # kg CO_2 für recycling

    # logistisches Wachstum
    args['k'] = 10**(-8)                     # r-Wert für logistisches Wachstum
    args['amax'] = 50*10**6               # Endwert für log Wachstum
    args['log_0'] = 100                         # Anfangszahl von e-Autos bei exp. growth

# t=np.linspace(0,10.,1000000)
# plt.plot(t,5.6-0.932*t- 0.02988 *t**2)
# f = np.array([t,np.cumsum(5.6-0.932*t- 0.02988 *t**2)])
# a= f[:,f[1,:]<0]
# a[0,0]
# np.sum(f)


    # Aus vorherigem Dokument
    args['a_0'] = 100                     # Anfangszahl von e-Autos bei exp. growth
    args['r'] = 0.428                     # growth rate
    args['c'] = 1*10**6                       # Zubaurate Units pro Zeit - lin

    # Grundgerüst
    t = np.linspace(-3,60,args['n'])    # Zeitenarray
    dx = (t[-1]-t[0])/(len(t)-1)

    scenario = 1
    which = int(input('1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr'))
     # 1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr,

    fkt = single_simple_real   #_delta, _block
    growth = growth_log         #_lin, _exp

    if which != 11 and which !=12:
        # Erstellen der Plotumgebung
        fig = plt.figure(figsize=(15,8))

        # Nutzen von einem Subplots
        ax = fig.add_subplot(1, 1, 1)

    # # RadioButton
    # axcolor = 'lightgoldenrodyellow'
    # rax = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
    # radio = RadioButtons(rax, ('1', '2','3','4','5','6','7','8','9','10'))
    # radio.on_clicked(func)

    # Achsenbereiche festlegen und Beschriftungen setzen
    if which==1:
        """Plot für SINGLE FUNCTION"""
        werte = fkt(t, args)
        fig.suptitle(r'Net single emission rate NSER $f(t,t_p)$', size = 20)            # Titel fuer Plots
        ax.set_xlim(t[0]+2020, 15+2020)                            # Achsenlimits x-Achse
        ax.set_ylim(-6, 14)                             # Achsenlimits y-Achse
        ax.set_xticks(np.arange(2017, 2036, 3))
        ax.set_yticks(np.arange(-4, 16, 2))
        ax.set_ylabel(r'Net $CO_2$ emission rate in $\frac{tons\ CO_2}{year}$', size = 20)   # Beschriftung y-Achse
        ax.set_xlabel(r'Time $t$ in years', size = 20)                # Beschriftung x-Achse
        ax.plot(t+2020, werte/1000, linewidth = 3)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.annotate(s='', xy=(2020,0), xytext=(2030,0), arrowprops=dict(arrowstyle='<->'))
        ax.annotate(s='', xy=(2029.5,0), xytext=(2029.5,-3.59), arrowprops=dict(arrowstyle='<->'))
        ax.annotate(s='', xy=(2020.5,7), xytext=(2020.25,7), arrowprops=dict(arrowstyle='-'))
        ellipse = matplotlib.patches.Ellipse(xy=(2020,7), width=.5, height=.3, angle=0, edgecolor='red',facecolor='none')
        ax.add_patch(ellipse)
        ax.text(2029.5, -1.75, 'Net emission rate \n during usage \n e(t) ', horizontalalignment='right', verticalalignment='center', size = 15)
        ax.text(2020.5, 7, ' Net initial \n emission \n'' $\ a_\mathrm{i}$  ', horizontalalignment='left', verticalalignment='center', size = 15)
        ax.text(2024.5, 0.5, 'Lifetime $t_\mathrm{life}$', horizontalalignment='center', verticalalignment='center', size = 15)
        #ax.grid()
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/01NSEF.pdf")
        plt.show()

    elif which==2:
        """Plot für INTEGRAL"""
        F = dx*np.cumsum(fkt(t, args))
        fig.suptitle('Cumulative Emission single car F(t)')            # Titel fuer Plots
        ax.set_xlim(t[0], t[-1])                            # Achsenlimits x-Achse
        ax.set_ylim(1.2*np.min(F)/1000, 1.2*np.max(F)/1000)                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Gesamt $CO_2$ Ausstoss in t')   # Beschriftung y-Achse
        ax.set_xlabel(r'Zeit $t$ in Jahren')                # Beschriftung x-Achse
        ax.plot(t, F/1000)
        plt.show()

    elif which==3:
        """Plot für Wachstum"""
        v = growth(t, args)
        # fig.suptitle('Exponentielles Wachstum mit Wachstumsrate {0}'.format(args['r']))            # Titel fuer Plots
        fig.suptitle('Logistisches Wachstum mit Wachstumsrate {0}'.format(args['k']))            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax.set_xlim(t[0], t[-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(v))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax.set_ylabel(r'Zubau von EV')   # Beschriftung y-Achse
        ax.set_xlabel(r'Zeit $t$ in Jahren')                # Beschriftung x-Achse
        ax.plot(t, v)
        plt.show()

    elif which==4:
        """Plot für Faltung"""
        f = fkt(t,args)
        v = growth(t,args)
        g = faltung(v,f,t)
        fig.suptitle('Current net emissions g(t)')            # Titel fuer Plots
        ax.set_xlim(t[0], t[-1])                            # Achsenlimits x-Achse
        ax.set_ylim(-0.05, 1.2*np.max(g)/1000)                             # Achsenlimits y-Achse
        ax.set_ylabel(r'$CO_2$ Ausstoss in t')   # Beschriftung y-Achse
        ax.set_xlabel(r'Zeit $t$ in Jahren')                # Beschriftung x-Achse
        ax.plot(t, g/1000)
        plt.show()

    elif which==5:
        """Plot für Total Emissions"""
        f = fkt(t,args)
        F = dx*np.cumsum(fkt(t, args))
        v = growth(t,args)
        g = faltung(v,f,t)
        G = dx*np.cumsum(g)
        fig.suptitle('Total Emissions G(t)')            # Titel fuer Plots
        ax.set_xlim(t[0], t[-1])                            # Achsenlimits x-Achse
        ax.set_ylim(-0.1, 1.2*np.max(G)/1000)                             # Achsenlimits y-Achse
        ax.set_ylabel(r'$CO_2$ Ausstoss in t')   # Beschriftung y-Achse
        ax.set_xlabel(r'Zeit $t$ in Jahren')                # Beschriftung x-Achse
        ax.plot(t, G/1000)
        plt.show()
        print(amor_time(t,G))

    elif which==6:
        """Plot für r-Abhängigkeit"""
        data = []
        f = fkt(t,args)
        for i in np.arange(10):
            args['r'] = i/10*0.2
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
        for j in np.arange(20):
            args['r'] = 0.2+j/20*0.1
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
        data.append([1,0])
        data = np.array(data)
        fig.suptitle('Amortization time')
        ax.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years')   # Beschriftung y-Achse
        ax.set_xlabel(r'Rate r')                # Beschriftung x-Achse
        ax.plot(data[:,0],data[:,1])
        ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        plt.show()

    elif which==7:
        """Plot für k-Abhängigkeit"""
        data = []
        f = fkt(t,args)
        for i in np.arange(100):
            args['k'] = i*10**(-10)
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['k'], amor_time(t,G)])
        for i in np.arange(50):
            args['k'] = i*10**(-8)/5 + 10**(-8)
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['k'], amor_time(t,G)])
        # for i in np.arange(50):
        #     args['k'] = i/500000
        #     v = growth(t,args)
        #     g = faltung(v,f,t)
        #     G = dx*np.cumsum(g)
        #     data.append([args['k'], amor_time(t,G)])
        data = np.array(data)
        fig.suptitle('Amortization time')
        ax.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years')   # Beschriftung y-Achse
        ax.set_xlabel(r'Rate k')                # Beschriftung x-Achse
        ax.plot(data[:,0],data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax.axvline(8.79255838*10**(-9), c='r')
        plt.show()

    elif which==8:
        "plot lifekm"
        data = []
        for i in np.arange(100):
            args['lifekm'] = 100000+4000*i
            f = fkt(t,args)
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['lifekm'], amor_time(t,G)])
        data = np.array(data)
        fig.suptitle('Amortization time')
        ax.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years')   # Beschriftung y-Achse
        ax.set_xlabel(r'lifedistance in km')                # Beschriftung x-Achse
        ax.plot(data[:,0],data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax.axvline(8.79255838*10**(-9), c='r')
        plt.show()

    elif which==9:
        "data initial"
        data = []
        for i in np.arange(100):
            args['neu_ausstoß'] = 2000+80*i
            f = fkt(t,args)
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['neu_ausstoß'], amor_time(t,G)])
        data = np.array(data)
        fig.suptitle('Amortization time')
        ax.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years')   # Beschriftung y-Achse
        ax.set_xlabel(r'CO2 Ausstoß bei Produktion in kg')                # Beschriftung x-Achse
        ax.plot(data[:,0],data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        plt.show()

    elif which==10:
        "data CO2 neutral"
        data = []
        for i in np.arange(60):
            args['neutral'] = 2021+i
            f = fkt(t,args)
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['neutral'], amor_time(t,G)])
        data = np.array(data)
        fig.suptitle('Amortization time')
        ax.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax.set_ylim(0, np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years')   # Beschriftung y-Achse
        ax.set_xlabel(r'Jahr in dem Strommix klimaneutral ist')                # Beschriftung x-Achse
        ax.plot(data[:,0],data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        plt.show()
    elif which == 11:
        b=2020
        args['k'] = 9.38*10**(-9)                     # r-Wert für logistisches Wachstum
        args['a_0'] = 136617                     # Anfangszahl von e-Autos bei exp. growth

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,8))
        #fig.suptitle('Different growth scenarios', size= 20)
        ax1.set_title('Linear \n', size = 25)
        ax1.plot(t+b, growth_lin(t, args))
        ax1.set_xlabel(r'Time $t$ in Years', size = 20)
        ax2.plot(t+b, growth_exp(t, args))
        ax2.set_xlabel(r'Time $t$ in Years', size = 20)
        ax2.set_title('Exponential \n', size = 25)

        ax3.plot(t+b, growth_log(t, args))
        ax3.set_xlabel(r'Time $t$ in Years', size = 20)
        ax3.set_title('Logistic \n', size = 25)

        ax4.plot(t+b, dx*np.cumsum(growth_lin(t, args)))
        ax4.set_xlabel(r'Time $t$ in Years', size = 20)
        ax5.plot(t+b, dx*np.cumsum(growth_exp(t, args)))
        ax5.set_xlabel(r'Time $t$ in Years', size = 20)
        ax6.plot(t+b, dx*np.cumsum(growth_log(t, args)))
        ax6.set_xlabel(r'Time $t$ in Years', size = 20)
        ax1.set_xlim(2020,2050)                            # Achsenlimits x-Achse
        ax2.set_xlim(2020,2050)                            # Achsenlimits x-Achse
        ax3.set_xlim(2020,2050)                            # Achsenlimits x-Achse
        ax4.set_xlim(2020,2050)                            # Achsenlimits x-Achse
        ax5.set_xlim(2020,2050)                            # Achsenlimits x-Achse
        ax6.set_xlim(2020,2050)                            # Achsenlimits x-Achse

        ax1.set_ylim(0,70*10**5)                            # Achsenlimits x-Achse
        ax2.set_ylim(0,70*10**5)                            # Achsenlimits x-Achse
        ax3.set_ylim(0,70*10**5)                            # Achsenlimits x-Achse
        ax4.set_ylim(0,70*10**6)                            # Achsenlimits x-Achse
        ax5.set_ylim(0,70*10**6)                            # Achsenlimits x-Achse
        ax6.set_ylim(0,70*10**6)                            # Achsenlimits x-Achse
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax3.tick_params(axis='both', which='major', labelsize=15)
        ax4.tick_params(axis='both', which='major', labelsize=15)
        ax5.tick_params(axis='both', which='major', labelsize=15)
        ax6.tick_params(axis='both', which='major', labelsize=15)

        ax1.set_ylabel('Growth rate \n'r'$\nu (t)$', size = 20)
    #    ax2.set_ylabel(r'# of Vehicles produced', size = 15)
    #    ax3.set_ylabel(r'# of Vehicles produced', size = 15)
        ax4.set_ylabel('Total number of EVs \n'r'$N^{\mathrm{tot}}(t)$', size = 20)
        #ax5.set_ylabel(r'Total amount of Vehicles', size = 15)
        #ax6.set_ylabel(r'Total amount of Vehicles', size = 15)
        ax1.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =15)
        ax2.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =15)
        ax3.text(0.05, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =15)
        ax4.text(0.05, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =15)
        ax5.text(0.05, 0.93, 'e)', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,size =15)
        ax6.text(0.05, 0.93, 'f)', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes,size =15)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/011compgrowth.pdf")

    elif which == 12:
        b=2020
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize= (12,7))
        werte = fkt(t, args)+fkt(t-1, args)+fkt(t-2, args)+fkt(t-3, args)+fkt(t-4, args)+fkt(t-5, args)+fkt(t-6, args)+fkt(t-7, args)+fkt(t-8, args)+fkt(t-9, args)
        werte2 = fkt(t,args)

        #fig.suptitle('single Emission funktion f(t)')            # Titel fuer Plots
        ax1.set_title('Single Vehicle \n', size = 15)
        ax1.set_xlim(t[0]+b, 14+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(-6, 14)                             # Achsenlimits y-Achse
        ax1.set_xticks(np.arange(b, 16+b, 5))
        ax1.set_yticks(np.arange(-30, 16, 10))
        ax1.set_ylabel('Net $CO_2$ emission rate\n''g(t) in 'r'$\frac{tons\ CO_2}{year}$', size = 15)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size = 15)                # Beschriftung x-Achse
        ax1.plot(t+b, werte2/1000, linewidth = 2)
        #ax1.grid()

        ax2.set_title('Multiple Vehicles \n', size = 15)
        ax2.set_xlim(t[0]+b, 14+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(-34, 14)                             # Achsenlimits y-Achse
        ax2.set_xticks(np.arange(b, 16+b, 5))
        ax2.set_yticks(np.arange(-30, 16, 10))
        ax2.set_ylabel('Net $CO_2$ emission rate\n''g(t) in 'r'$\frac{tons\ CO_2}{year}$', size = 15)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size = 15)                # Beschriftung x-Achse
        ax2.plot(t+b, werte/1000, linewidth = 2)
        #ax2.grid()

        ax3.set_xlim(t[0]+b, 14+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(-6, 14)                             # Achsenlimits y-Achse
        ax3.set_xticks(np.arange(b, 16+b, 5))
        ax3.set_yticks(np.arange(-20, 31, 10))
        ax3.set_ylabel('Net total $CO_2$ emission\n''G(t) in $tons\ CO_2$', size = 15)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = 15)                # Beschriftung x-Achse
        ax3.plot(t+b, dx*np.cumsum(werte2)/1000, linewidth = 2)
        #ax3.grid()

        ax4.set_xlim(t[0]+b, 14+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(-18, 30)                             # Achsenlimits y-Achse
        ax4.set_xticks(np.arange(b, 16+b, 5))
        ax4.set_yticks(np.arange(-20, 31, 10))
        ax4.set_ylabel('Net total $CO_2$ emission\n''G(t) in $tons\ CO_2$', size = 15)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = 15)                # Beschriftung x-Achse
        ax4.plot(t+b, dx*np.cumsum(werte)/1000, linewidth = 2)
        #ax4.grid()

        ax1.axhline(linewidth=1, linestyle= '--', color="black")        # inc. width of y-axis and color it red
        ax2.axhline(linewidth=1, linestyle= '--', color="black")        # inc. width of y-axis and color it red
        ax3.axhline(linewidth=1, linestyle= '--', color="black")        # inc. width of y-axis and color it red
        ax4.axhline(linewidth=1, linestyle= '--', color="black")        # inc. width of y-axis and color it red
        ax3.annotate(r"$\Delta t_{a} = 3.7$ years", xy=(b+3.7, 0), xytext=(2025.4, 5),arrowprops=dict(arrowstyle="->"), size = 10)
        ax4.annotate(r"$\Delta t_{a} \approx 7$ years", xy=(b+6.8, 0), xytext=(2028.4, 10),arrowprops=dict(arrowstyle="->"), size = 10)

        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        ax1.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =15)
        ax2.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =15)
        ax3.text(0.05, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =15)
        ax4.text(0.05, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =15)

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/011semian.pdf")
        plt.show()

    plt.draw()




if __name__ == "__main__":
    main()
