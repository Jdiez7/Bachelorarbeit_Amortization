import numpy as np                          # Modul zur allgemeinen Berechnung.
from matplotlib import pyplot as plt           # Graphikbefehle.def main():
from scipy import optimize

def test_func2(x, a, b):
    return a*x+b

def test_func(x, c, xhalb):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    amax = 47.7*10**6
    return amax/(1+np.exp(-c*(x-xhalb)))

def main():


#    x = np.array([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 ,2020])
    x = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    y = np.array([2307, 4541, 7114, 12156, 18948, 25502, 34022, 53861, 83175, 136617])
    x2=np.linspace(2000,2050, 1000)
    size = 18
    headsize = 23
    params, params_covariance = optimize.curve_fit(test_func, x, y, p0=[0.4, 2030]) # 29-1000 sind die interessanten Messwerte

    # Erstellen der Plotumgebung
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,7))
    plt.subplots_adjust(hspace= 0.6)
    """Plot 1 - exponentiell """
    ax1.scatter(x, y, 20, label='Data')
    ax1.plot(x2, test_func(x2, params[0], params[1]), 'r--',label='Fit')
    #ax.plot(x2, 34022*np.exp(0.439*x2))
    ax1.set_xlabel('Year',size=size)
    ax1.set_ylabel(r'Amount of EVs',size=size)
    ax1.set_ylim(10**3, 10**6)
    ax1.set_yscale('symlog')
    ax1.set_xlim(2010, 2020)
    #ax.grid(b=None, which='major', axis='both', color='grey', linestyle='-',linewidth=0.5)
    ax1.legend()
    ax1.set_title('Fit for logistic increase',size=headsize)
    """Plot 2 - normal"""
    ax2.scatter(x, y, 20, label='Data')
    ax2.plot(x2, test_func(x2, params[0], params[1]), 'r--',label='Fit')
    #ax.plot(x2, 34022*np.exp(0.439*x2))
    #ax1.plot(x2, np.cumsum((x2[1]-x2[0])*47.7*10**6*params[0]*np.exp(-params[0]*(x2-params[1]))/((1+np.exp(-params[0]*(x2-(params[1]))))**2)))
    ax2.set_xlabel('Year',size=size)
    ax2.set_ylabel(r'Amount of EVs',size=size)
    ax2.set_ylim(10**3, 0.6*10**8)
    ax2.set_xlim(2010, 2050)
    ax2.set_xticks(np.arange(2010,2051,10))
    ax2.hlines(4.77*10**7,2010,2050, 'k', ls='--', label='total # of vehicles')
    ax2.set_title('Development until 2050',size=headsize)
    ax2.vlines(params[1], 0, 4.77/2*10**7, ls=':',label=r'$t_{1/2}$')
    ax2.legend()
    ax1.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
    ax2.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/011fitlog.pdf")


    plt.show()
    print(params, params_covariance)



if __name__ == "__main__":
    main()
