#ifndef UMUQ_SPECIESNAME_H
#define UMUQ_SPECIESNAME_H

#include <iostream>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>

#include "misc/parser.hpp"

namespace umuq
{

/*! 
 * \defgroup Species_Module Species module
 *
 * This is the species Module of UMUQ providing all necessary classes 
 * for species from the periodic table and user specified species currently 
 * supported in UMUQ.
 * 
 * \note User defined species are for compatibility with [OpenKIM](https://openkim.org/) 
 */

/*!
 * \enum SpeciesID
 * \ingroup Species_Module
 * 
 * \brief Different species name 
 * 
 * \note User defined species are for compatibility with [OpenKIM](https://openkim.org/) 
 * 
 */
enum class SpeciesID : int
{
	/*! \brief The \c electron species */
	electron,
	/*! \brief The \c Hydrogen species */
	H,
	/*! \brief The \c Helium species */
	He,
	/*! \brief The \c Lithium species */
	Li,
	/*! \brief The \c Beryllium species */
	Be,
	/*! \brief The \c Boron species */
	B,
	/*! \brief The \c Carbon species */
	C,
	/*! \brief The \c Nitrogen species */
	N,
	/*! \brief The \c Oxygen species */
	O,
	/*! \brief The \c Fluorine species */
	F,
	/*! \brief The \c Neon species */
	Ne,
	/*! \brief The \c Sodium species */
	Na,
	/*! \brief The \c Magnesium species */
	Mg,
	/*! \brief The \c Aluminum species */
	Al,
	/*! \brief The \c Silicon species */
	Si,
	/*! \brief The \c Phosphorus species */
	P,
	/*! \brief The \c Sulfur species */
	S,
	/*! \brief The \c Chlorine species */
	Cl,
	/*! \brief The \c Argon species */
	Ar,
	/*! \brief The \c Potassium species */
	K,
	/*! \brief The \c Calcium species */
	Ca,
	/*! \brief The \c Scandium species */
	Sc,
	/*! \brief The \c Titanium species */
	Ti,
	/*! \brief The \c Vanadium species */
	V,
	/*! \brief The \c Chromium species */
	Cr,
	/*! \brief The \c Manganese species */
	Mn,
	/*! \brief The \c Iron species */
	Fe,
	/*! \brief The \c Cobalt species */
	Co,
	/*! \brief The \c Nickel species */
	Ni,
	/*! \brief The \c Copper species */
	Cu,
	/*! \brief The \c Zinc species */
	Zn,
	/*! \brief The \c Gallium species */
	Ga,
	/*! \brief The \c Germanium species */
	Ge,
	/*! \brief The \c Arsenic species */
	As,
	/*! \brief The \c Selenium species */
	Se,
	/*! \brief The \c Bromine species */
	Br,
	/*! \brief The \c Krypton species */
	Kr,
	/*! \brief The \c Rubidium species */
	Rb,
	/*! \brief The \c Strontium species */
	Sr,
	/*! \brief The \c Yttrium species */
	Y,
	/*! \brief The \c Zirconium species */
	Zr,
	/*! \brief The \c Niobium species */
	Nb,
	/*! \brief The \c Molybdenum species */
	Mo,
	/*! \brief The \c Technetium species */
	Tc,
	/*! \brief The \c Ruthenium species */
	Ru,
	/*! \brief The \c Rhodium species */
	Rh,
	/*! \brief The \c Palladium species */
	Pd,
	/*! \brief The \c Silver species */
	Ag,
	/*! \brief The \c Cadmium species */
	Cd,
	/*! \brief The \c Indium species */
	In,
	/*! \brief The \c Tin species */
	Sn,
	/*! \brief The \c Antimony species */
	Sb,
	/*! \brief The \c Tellurium species */
	Te,
	/*! \brief The \c Iodine species */
	I,
	/*! \brief The \c Xenon species */
	Xe,
	/*! \brief The \c Cesium species */
	Cs,
	/*! \brief The \c Barium species */
	Ba,
	/*! \brief The \c Lanthanum species */
	La,
	/*! \brief The \c Cerium species */
	Ce,
	/*! \brief The \c Praseodymium species */
	Pr,
	/*! \brief The \c Neodymium species */
	Nd,
	/*! \brief The \c Promethium species */
	Pm,
	/*! \brief The \c Samarium species */
	Sm,
	/*! \brief The \c Europium species */
	Eu,
	/*! \brief The \c Gadolinium species */
	Gd,
	/*! \brief The \c Terbium species */
	Tb,
	/*! \brief The \c Dysprosium species */
	Dy,
	/*! \brief The \c Holmium species */
	Ho,
	/*! \brief The \c Erbium species */
	Er,
	/*! \brief The \c Thulium species */
	Tm,
	/*! \brief The \c Ytterbium species */
	Yb,
	/*! \brief The \c Lutetium species */
	Lu,
	/*! \brief The \c Hafnium species */
	Hf,
	/*! \brief The \c Tantalum species */
	Ta,
	/*! \brief The \c Tungsten species */
	W,
	/*! \brief The \c Rhenium species */
	Re,
	/*! \brief The \c Osmium species */
	Os,
	/*! \brief The \c Iridium species */
	Ir,
	/*! \brief The \c Platinum species */
	Pt,
	/*! \brief The \c Gold species */
	Au,
	/*! \brief The \c Mercury species */
	Hg,
	/*! \brief The \c Thallium species */
	Tl,
	/*! \brief The \c Lead species */
	Pb,
	/*! \brief The \c Bismuth species */
	Bi,
	/*! \brief The \c Polonium species */
	Po,
	/*! \brief The \c Astatine species */
	At,
	/*! \brief The \c Radon species */
	Rn,
	/*! \brief The \c Francium species */
	Fr,
	/*! \brief The \c Radium species */
	Ra,
	/*! \brief The \c Actinium species */
	Ac,
	/*! \brief The \c Thorium species */
	Th,
	/*! \brief The \c Protactinium species */
	Pa,
	/*! \brief The \c Uranium species */
	U,
	/*! \brief The \c Neptunium species */
	Np,
	/*! \brief The \c Plutonium species */
	Pu,
	/*! \brief The \c Americium species */
	Am,
	/*! \brief The \c Curium species */
	Cm,
	/*! \brief The \c Berkelium species */
	Bk,
	/*! \brief The \c Californium species */
	Cf,
	/*! \brief The \c Einsteinium species */
	Es,
	/*! \brief The \c Fermium species */
	Fm,
	/*! \brief The \c Mendelevium species */
	Md,
	/*! \brief The \c Nobelium species */
	No,
	/*! \brief The \c Lawrencium species */
	Lr,
	/*! \brief The \c Rutherfordium species */
	Rf,
	/*! \brief The \c Dubnium species */
	Db,
	/*! \brief The \c Seaborgium species */
	Sg,
	/*! \brief The \c Bohrium species */
	Bh,
	/*! \brief The \c Hassium species */
	Hs,
	/*! \brief The \c Meitnerium species */
	Mt,
	/*! \brief The \c Darmstadtium species */
	Ds,
	/*! \brief The \c Roentgenium species */
	Rg,
	/*! \brief The \c Copernicium species */
	Cn,
	/*! \brief The \c Ununtrium species */
	Uut,
	/*! \brief The \c Flerovium species */
	Fl,
	/*! \brief The \c Ununpentium species */
	Uup,
	/*! \brief The \c Livermorium species */
	Lv,
	/*! \brief The \c Ununseptium species */
	Uus,
	/*! \brief The \c Ununoctium species */
	Uuo,
	/*! \brief The \c user defined species */
	user01,
	/*! \brief The \c user defined species */
	user02,
	/*! \brief The \c user defined species */
	user03,
	/*! \brief The \c user defined species */
	user04,
	/*! \brief The \c user defined species */
	user05,
	/*! \brief The \c user defined species */
	user06,
	/*! \brief The \c user defined species */
	user07,
	/*! \brief The \c user defined species */
	user08,
	/*! \brief The \c user defined species */
	user09,
	/*! \brief The \c user defined species */
	user10,
	/*! \brief The \c user defined species */
	user11,
	/*! \brief The \c user defined species */
	user12,
	/*! \brief The \c user defined species */
	user13,
	/*! \brief The \c user defined species */
	user14,
	/*! \brief The \c user defined species */
	user15,
	/*! \brief The \c user defined species */
	user16,
	/*! \brief The \c user defined species */
	user17,
	/*! \brief The \c user defined species */
	user18,
	/*! \brief The \c user defined species */
	user19,
	/*! \brief The \c user defined species */
	user20
};

/*! \class speciesAttribute
 * \ingroup Species_Module
 * 
 * \brief A class of common attributes possessed by all species
 * 
 */
struct speciesAttribute
{
	/*!
     * \brief Construct a new species object
     */
	speciesAttribute();

	/*!
     * \brief Construct a new species object
     * 
     * \param SpeciesName  Species name corresponding to the provided string
     */
	speciesAttribute(std::string const &SpeciesName);

	/*!
     * \brief Construct a new species object
     * 
     * \param SpeciesName  Species name corresponding to the provided string
     * \param Mass         The specified mass
     */
	speciesAttribute(std::string const &SpeciesName, double const Mass);

	/*! \brief Species name */
	std::string name;

	/*! \brief Species mass */
	double mass;
};

speciesAttribute::speciesAttribute() : name("unknown"), mass(std::numeric_limits<double>::infinity()) {}
speciesAttribute::speciesAttribute(std::string const &SpeciesName) : name(SpeciesName), mass(std::numeric_limits<double>::infinity()) {}
speciesAttribute::speciesAttribute(std::string const &SpeciesName, double const Mass) : name(SpeciesName), mass(Mass) {}

/*! \class species
 * \ingroup Species_Module
 * 
 * \brief 
 * 
 */
class species
{
  public:
	/*!
     * \brief Construct a new species object
     * 
     */
	species();

	/*!
     * \brief Destroy the species object
     * 
     */
	~species();

	/*!
     * \brief Get the Number Of Species 
     * 
     * \returns int Get the number of standard species defined
     */
	inline int getNumberOfSpecies();

	/*!
     * \brief Get the Species Name 
     * 
     * Get the identity of each defined standard species.
     * 
     * 
     * \param index  Zero-based index uniquely labeling each defined standard               
     *               speciesName. \sa SpeciesID
     * 
     * \returns std::string The speciesName object associated with the input index \sa SpeciesID
     *                      If the input index `< 0` or index ` >= ` number of species, it returns 
     * 						`unknown`
     */
	inline std::string getSpeciesName(int const index);

	/*!
     * \brief Get the Species Name 
     * 
     * Get the identity of each defined standard species.
     * 
     * 
     * \param index  Zero-based index uniquely labeling each defined standard               
     *               speciesName. \sa SpeciesID
     * 
     * \returns std::string The speciesName object associated with the input index \sa SpeciesID
     *                      If the input index `< 0` or index ` >= ` number of species, it returns 
     * 						`unknown`
     */
	inline std::string getSpeciesName(SpeciesID const index);

	/*!
     * \brief Get the Species ID
     * 
     * \param SpeciesName The species name
     * 
     * \return int Zero-based index uniquely labeling each defined standard species name.
     */
	int getSpeciesID(std::string const &SpeciesName);

	/*!
     * \brief Get the speciesAttribute object
     * 
     * \param index  Zero-based index uniquely labeling each defined standard               
     *               speciesName. \sa SpeciesID
     * 
     * \return speciesAttribute A speciesAttribute object
     */
	inline speciesAttribute getSpecies(int const index);

	/*!
     * \brief Get the speciesAttribute object
     * 
     * \param index Zero-based index uniquely labeling each defined standard               
     *               speciesName. \sa SpeciesID
     * 
     * \return speciesAttribute A speciesAttribute object
     */
	inline speciesAttribute getSpecies(SpeciesID const index);

	/*!
     * \brief Get the speciesAttribute object
     * 
     * \param SpeciesName   The species name
     * 
     * \return speciesAttribute  A speciesAttribute object
     */
	speciesAttribute getSpecies(std::string const &SpeciesName);

  private:
	/*! A sorted associative container that contains species names pairs with species ID. */
	std::map<std::string const, SpeciesID const> speciesMap;

	/*! Vector of species. */
	std::vector<speciesAttribute> speciesTable;
};

species::species()
{
	speciesMap = std::map<std::string const, SpeciesID const>{
		{"electron", SpeciesID::electron},
		{"H", SpeciesID::H},
		{"He", SpeciesID::He},
		{"Li", SpeciesID::Li},
		{"Be", SpeciesID::Be},
		{"B", SpeciesID::B},
		{"C", SpeciesID::C},
		{"N", SpeciesID::N},
		{"O", SpeciesID::O},
		{"F", SpeciesID::F},
		{"Ne", SpeciesID::Ne},
		{"Na", SpeciesID::Na},
		{"Mg", SpeciesID::Mg},
		{"Al", SpeciesID::Al},
		{"Si", SpeciesID::Si},
		{"P", SpeciesID::P},
		{"S", SpeciesID::S},
		{"Cl", SpeciesID::Cl},
		{"Ar", SpeciesID::Ar},
		{"K", SpeciesID::K},
		{"Ca", SpeciesID::Ca},
		{"Sc", SpeciesID::Sc},
		{"Ti", SpeciesID::Ti},
		{"V", SpeciesID::V},
		{"Cr", SpeciesID::Cr},
		{"Mn", SpeciesID::Mn},
		{"Fe", SpeciesID::Fe},
		{"Co", SpeciesID::Co},
		{"Ni", SpeciesID::Ni},
		{"Cu", SpeciesID::Cu},
		{"Zn", SpeciesID::Zn},
		{"Ga", SpeciesID::Ga},
		{"Ge", SpeciesID::Ge},
		{"As", SpeciesID::As},
		{"Se", SpeciesID::Se},
		{"Br", SpeciesID::Br},
		{"Kr", SpeciesID::Kr},
		{"Rb", SpeciesID::Rb},
		{"Sr", SpeciesID::Sr},
		{"Y", SpeciesID::Y},
		{"Zr", SpeciesID::Zr},
		{"Nb", SpeciesID::Nb},
		{"Mo", SpeciesID::Mo},
		{"Tc", SpeciesID::Tc},
		{"Ru", SpeciesID::Ru},
		{"Rh", SpeciesID::Rh},
		{"Pd", SpeciesID::Pd},
		{"Ag", SpeciesID::Ag},
		{"Cd", SpeciesID::Cd},
		{"In", SpeciesID::In},
		{"Sn", SpeciesID::Sn},
		{"Sb", SpeciesID::Sb},
		{"Te", SpeciesID::Te},
		{"I", SpeciesID::I},
		{"Xe", SpeciesID::Xe},
		{"Cs", SpeciesID::Cs},
		{"Ba", SpeciesID::Ba},
		{"La", SpeciesID::La},
		{"Ce", SpeciesID::Ce},
		{"Pr", SpeciesID::Pr},
		{"Nd", SpeciesID::Nd},
		{"Pm", SpeciesID::Pm},
		{"Sm", SpeciesID::Sm},
		{"Eu", SpeciesID::Eu},
		{"Gd", SpeciesID::Gd},
		{"Tb", SpeciesID::Tb},
		{"Dy", SpeciesID::Dy},
		{"Ho", SpeciesID::Ho},
		{"Er", SpeciesID::Er},
		{"Tm", SpeciesID::Tm},
		{"Yb", SpeciesID::Yb},
		{"Lu", SpeciesID::Lu},
		{"Hf", SpeciesID::Hf},
		{"Ta", SpeciesID::Ta},
		{"W", SpeciesID::W},
		{"Re", SpeciesID::Re},
		{"Os", SpeciesID::Os},
		{"Ir", SpeciesID::Ir},
		{"Pt", SpeciesID::Pt},
		{"Au", SpeciesID::Au},
		{"Hg", SpeciesID::Hg},
		{"Tl", SpeciesID::Tl},
		{"Pb", SpeciesID::Pb},
		{"Bi", SpeciesID::Bi},
		{"Po", SpeciesID::Po},
		{"At", SpeciesID::At},
		{"Rn", SpeciesID::Rn},
		{"Fr", SpeciesID::Fr},
		{"Ra", SpeciesID::Ra},
		{"Ac", SpeciesID::Ac},
		{"Th", SpeciesID::Th},
		{"Pa", SpeciesID::Pa},
		{"U", SpeciesID::U},
		{"Np", SpeciesID::Np},
		{"Pu", SpeciesID::Pu},
		{"Am", SpeciesID::Am},
		{"Cm", SpeciesID::Cm},
		{"Bk", SpeciesID::Bk},
		{"Cf", SpeciesID::Cf},
		{"Es", SpeciesID::Es},
		{"Fm", SpeciesID::Fm},
		{"Md", SpeciesID::Md},
		{"No", SpeciesID::No},
		{"Lr", SpeciesID::Lr},
		{"Rf", SpeciesID::Rf},
		{"Db", SpeciesID::Db},
		{"Sg", SpeciesID::Sg},
		{"Bh", SpeciesID::Bh},
		{"Hs", SpeciesID::Hs},
		{"Mt", SpeciesID::Mt},
		{"Ds", SpeciesID::Ds},
		{"Rg", SpeciesID::Rg},
		{"Cn", SpeciesID::Cn},
		{"Uut", SpeciesID::Uut},
		{"Fl", SpeciesID::Fl},
		{"Uup", SpeciesID::Uup},
		{"Lv", SpeciesID::Lv},
		{"Uus", SpeciesID::Uus},
		{"Uuo", SpeciesID::Uuo},
		{"user01", SpeciesID::user01},
		{"user02", SpeciesID::user02},
		{"user03", SpeciesID::user03},
		{"user04", SpeciesID::user04},
		{"user05", SpeciesID::user05},
		{"user06", SpeciesID::user06},
		{"user07", SpeciesID::user07},
		{"user08", SpeciesID::user08},
		{"user09", SpeciesID::user09},
		{"user10", SpeciesID::user10},
		{"user11", SpeciesID::user11},
		{"user12", SpeciesID::user12},
		{"user13", SpeciesID::user13},
		{"user14", SpeciesID::user14},
		{"user15", SpeciesID::user15},
		{"user16", SpeciesID::user16},
		{"user17", SpeciesID::user17},
		{"user18", SpeciesID::user18},
		{"user19", SpeciesID::user19},
		{"user20", SpeciesID::user20}};
	speciesTable.resize(speciesMap.size());
	speciesTable[static_cast<int>(SpeciesID::electron)] = std::move(speciesAttribute("electron"));
	speciesTable[static_cast<int>(SpeciesID::H)] = std::move(speciesAttribute("H", 1.00794));
	speciesTable[static_cast<int>(SpeciesID::He)] = std::move(speciesAttribute("He", 4.00260));
	speciesTable[static_cast<int>(SpeciesID::Li)] = std::move(speciesAttribute("Li", 6.941));
	speciesTable[static_cast<int>(SpeciesID::Be)] = std::move(speciesAttribute("Be", 9.01218));
	speciesTable[static_cast<int>(SpeciesID::B)] = std::move(speciesAttribute("B", 10.811));
	speciesTable[static_cast<int>(SpeciesID::C)] = std::move(speciesAttribute("C", 12.0107));
	speciesTable[static_cast<int>(SpeciesID::N)] = std::move(speciesAttribute("N", 14.00674));
	speciesTable[static_cast<int>(SpeciesID::O)] = std::move(speciesAttribute("O", 15.9994));
	speciesTable[static_cast<int>(SpeciesID::F)] = std::move(speciesAttribute("F", 18.9884));
	speciesTable[static_cast<int>(SpeciesID::Ne)] = std::move(speciesAttribute("Ne", 20.1797));
	speciesTable[static_cast<int>(SpeciesID::Na)] = std::move(speciesAttribute("Na", 22.98977));
	speciesTable[static_cast<int>(SpeciesID::Mg)] = std::move(speciesAttribute("Mg", 24.3050));
	speciesTable[static_cast<int>(SpeciesID::Al)] = std::move(speciesAttribute("Al", 26.98154));
	speciesTable[static_cast<int>(SpeciesID::Si)] = std::move(speciesAttribute("Si", 28.0855));
	speciesTable[static_cast<int>(SpeciesID::P)] = std::move(speciesAttribute("P", 30.97376));
	speciesTable[static_cast<int>(SpeciesID::S)] = std::move(speciesAttribute("S", 32.066));
	speciesTable[static_cast<int>(SpeciesID::Cl)] = std::move(speciesAttribute("Cl", 35.4527));
	speciesTable[static_cast<int>(SpeciesID::Ar)] = std::move(speciesAttribute("Ar", 39.948));
	speciesTable[static_cast<int>(SpeciesID::K)] = std::move(speciesAttribute("K", 39.0983));
	speciesTable[static_cast<int>(SpeciesID::Ca)] = std::move(speciesAttribute("Ca", 40.078));
	speciesTable[static_cast<int>(SpeciesID::Sc)] = std::move(speciesAttribute("Sc", 44.95591));
	speciesTable[static_cast<int>(SpeciesID::Ti)] = std::move(speciesAttribute("Ti", 47.867));
	speciesTable[static_cast<int>(SpeciesID::V)] = std::move(speciesAttribute("V", 50.9415));
	speciesTable[static_cast<int>(SpeciesID::Cr)] = std::move(speciesAttribute("Cr", 51.9961));
	speciesTable[static_cast<int>(SpeciesID::Mn)] = std::move(speciesAttribute("Mn", 54.93805));
	speciesTable[static_cast<int>(SpeciesID::Fe)] = std::move(speciesAttribute("Fe", 55.845));
	speciesTable[static_cast<int>(SpeciesID::Co)] = std::move(speciesAttribute("Co", 58.9332));
	speciesTable[static_cast<int>(SpeciesID::Ni)] = std::move(speciesAttribute("Ni", 58.6934));
	speciesTable[static_cast<int>(SpeciesID::Cu)] = std::move(speciesAttribute("Cu", 63.546));
	speciesTable[static_cast<int>(SpeciesID::Zn)] = std::move(speciesAttribute("Zn", 65.39));
	speciesTable[static_cast<int>(SpeciesID::Ga)] = std::move(speciesAttribute("Ga", 69.723));
	speciesTable[static_cast<int>(SpeciesID::Ge)] = std::move(speciesAttribute("Ge", 72.61));
	speciesTable[static_cast<int>(SpeciesID::As)] = std::move(speciesAttribute("As", 74.9216));
	speciesTable[static_cast<int>(SpeciesID::Se)] = std::move(speciesAttribute("Se", 78.96));
	speciesTable[static_cast<int>(SpeciesID::Br)] = std::move(speciesAttribute("Br", 79.904));
	speciesTable[static_cast<int>(SpeciesID::Kr)] = std::move(speciesAttribute("Kr", 83.80));
	speciesTable[static_cast<int>(SpeciesID::Rb)] = std::move(speciesAttribute("Rb", 85.4678));
	speciesTable[static_cast<int>(SpeciesID::Sr)] = std::move(speciesAttribute("Sr", 87.62));
	speciesTable[static_cast<int>(SpeciesID::Y)] = std::move(speciesAttribute("Y", 88.90585));
	speciesTable[static_cast<int>(SpeciesID::Zr)] = std::move(speciesAttribute("Zr", 91.224));
	speciesTable[static_cast<int>(SpeciesID::Nb)] = std::move(speciesAttribute("Nb", 92.90638));
	speciesTable[static_cast<int>(SpeciesID::Mo)] = std::move(speciesAttribute("Mo", 95.94));
	speciesTable[static_cast<int>(SpeciesID::Tc)] = std::move(speciesAttribute("Tc", 98.0));
	speciesTable[static_cast<int>(SpeciesID::Ru)] = std::move(speciesAttribute("Ru", 101.07));
	speciesTable[static_cast<int>(SpeciesID::Rh)] = std::move(speciesAttribute("Rh", 102.9055));
	speciesTable[static_cast<int>(SpeciesID::Pd)] = std::move(speciesAttribute("Pd", 106.42));
	speciesTable[static_cast<int>(SpeciesID::Ag)] = std::move(speciesAttribute("Ag", 107.8682));
	speciesTable[static_cast<int>(SpeciesID::Cd)] = std::move(speciesAttribute("Cd", 112.411));
	speciesTable[static_cast<int>(SpeciesID::In)] = std::move(speciesAttribute("In", 114.818));
	speciesTable[static_cast<int>(SpeciesID::Sn)] = std::move(speciesAttribute("Sn", 118.710));
	speciesTable[static_cast<int>(SpeciesID::Sb)] = std::move(speciesAttribute("Sb", 121.760));
	speciesTable[static_cast<int>(SpeciesID::Te)] = std::move(speciesAttribute("Te", 127.60));
	speciesTable[static_cast<int>(SpeciesID::I)] = std::move(speciesAttribute("I", 126.90447));
	speciesTable[static_cast<int>(SpeciesID::Xe)] = std::move(speciesAttribute("Xe", 131.29));
	speciesTable[static_cast<int>(SpeciesID::Cs)] = std::move(speciesAttribute("Cs", 132.90545));
	speciesTable[static_cast<int>(SpeciesID::Ba)] = std::move(speciesAttribute("Ba", 137.327));
	speciesTable[static_cast<int>(SpeciesID::La)] = std::move(speciesAttribute("La", 138.9055));
	speciesTable[static_cast<int>(SpeciesID::Ce)] = std::move(speciesAttribute("Ce", 140.116));
	speciesTable[static_cast<int>(SpeciesID::Pr)] = std::move(speciesAttribute("Pr", 140.90765));
	speciesTable[static_cast<int>(SpeciesID::Nd)] = std::move(speciesAttribute("Nd", 144.24));
	speciesTable[static_cast<int>(SpeciesID::Pm)] = std::move(speciesAttribute("Pm", 145.0));
	speciesTable[static_cast<int>(SpeciesID::Sm)] = std::move(speciesAttribute("Sm", 150.36));
	speciesTable[static_cast<int>(SpeciesID::Eu)] = std::move(speciesAttribute("Eu", 151.964));
	speciesTable[static_cast<int>(SpeciesID::Gd)] = std::move(speciesAttribute("Gd", 157.25));
	speciesTable[static_cast<int>(SpeciesID::Tb)] = std::move(speciesAttribute("Tb", 158.92534));
	speciesTable[static_cast<int>(SpeciesID::Dy)] = std::move(speciesAttribute("Dy", 162.50));
	speciesTable[static_cast<int>(SpeciesID::Ho)] = std::move(speciesAttribute("Ho", 164.93032));
	speciesTable[static_cast<int>(SpeciesID::Er)] = std::move(speciesAttribute("Er", 167.26));
	speciesTable[static_cast<int>(SpeciesID::Tm)] = std::move(speciesAttribute("Tm", 168.93421));
	speciesTable[static_cast<int>(SpeciesID::Yb)] = std::move(speciesAttribute("Yb", 173.04));
	speciesTable[static_cast<int>(SpeciesID::Lu)] = std::move(speciesAttribute("Lu", 174.967));
	speciesTable[static_cast<int>(SpeciesID::Hf)] = std::move(speciesAttribute("Hf", 178.49));
	speciesTable[static_cast<int>(SpeciesID::Ta)] = std::move(speciesAttribute("Ta", 180.9479));
	speciesTable[static_cast<int>(SpeciesID::W)] = std::move(speciesAttribute("W", 183.84));
	speciesTable[static_cast<int>(SpeciesID::Re)] = std::move(speciesAttribute("Re", 186.207));
	speciesTable[static_cast<int>(SpeciesID::Os)] = std::move(speciesAttribute("Os", 190.23));
	speciesTable[static_cast<int>(SpeciesID::Ir)] = std::move(speciesAttribute("Ir", 192.217));
	speciesTable[static_cast<int>(SpeciesID::Pt)] = std::move(speciesAttribute("Pt", 195.078));
	speciesTable[static_cast<int>(SpeciesID::Au)] = std::move(speciesAttribute("Au", 196.96655));
	speciesTable[static_cast<int>(SpeciesID::Hg)] = std::move(speciesAttribute("Hg", 200.59));
	speciesTable[static_cast<int>(SpeciesID::Tl)] = std::move(speciesAttribute("Tl", 204.3833));
	speciesTable[static_cast<int>(SpeciesID::Pb)] = std::move(speciesAttribute("Pb", 207.2));
	speciesTable[static_cast<int>(SpeciesID::Bi)] = std::move(speciesAttribute("Bi", 208.98038));
	speciesTable[static_cast<int>(SpeciesID::Po)] = std::move(speciesAttribute("Po", 209.0));
	speciesTable[static_cast<int>(SpeciesID::At)] = std::move(speciesAttribute("At", 210.0));
	speciesTable[static_cast<int>(SpeciesID::Rn)] = std::move(speciesAttribute("Rn", 222.0));
	speciesTable[static_cast<int>(SpeciesID::Fr)] = std::move(speciesAttribute("Fr", 223.0));
	speciesTable[static_cast<int>(SpeciesID::Ra)] = std::move(speciesAttribute("Ra", 226.0));
	speciesTable[static_cast<int>(SpeciesID::Ac)] = std::move(speciesAttribute("Ac", 227.0));
	speciesTable[static_cast<int>(SpeciesID::Th)] = std::move(speciesAttribute("Th", 232.0381));
	speciesTable[static_cast<int>(SpeciesID::Pa)] = std::move(speciesAttribute("Pa", 231.03588));
	speciesTable[static_cast<int>(SpeciesID::U)] = std::move(speciesAttribute("U", 238.0289));
	speciesTable[static_cast<int>(SpeciesID::Np)] = std::move(speciesAttribute("Np", 237.0));
	speciesTable[static_cast<int>(SpeciesID::Pu)] = std::move(speciesAttribute("Pu", 244.0));
	speciesTable[static_cast<int>(SpeciesID::Am)] = std::move(speciesAttribute("Am"));
	speciesTable[static_cast<int>(SpeciesID::Cm)] = std::move(speciesAttribute("Cm"));
	speciesTable[static_cast<int>(SpeciesID::Bk)] = std::move(speciesAttribute("Bk"));
	speciesTable[static_cast<int>(SpeciesID::Cf)] = std::move(speciesAttribute("Cf"));
	speciesTable[static_cast<int>(SpeciesID::Es)] = std::move(speciesAttribute("Es"));
	speciesTable[static_cast<int>(SpeciesID::Fm)] = std::move(speciesAttribute("Fm"));
	speciesTable[static_cast<int>(SpeciesID::Md)] = std::move(speciesAttribute("Md"));
	speciesTable[static_cast<int>(SpeciesID::No)] = std::move(speciesAttribute("No"));
	speciesTable[static_cast<int>(SpeciesID::Lr)] = std::move(speciesAttribute("Lr"));
	speciesTable[static_cast<int>(SpeciesID::Rf)] = std::move(speciesAttribute("Rf"));
	speciesTable[static_cast<int>(SpeciesID::Db)] = std::move(speciesAttribute("Db"));
	speciesTable[static_cast<int>(SpeciesID::Sg)] = std::move(speciesAttribute("Sg"));
	speciesTable[static_cast<int>(SpeciesID::Bh)] = std::move(speciesAttribute("Bh"));
	speciesTable[static_cast<int>(SpeciesID::Hs)] = std::move(speciesAttribute("Hs"));
	speciesTable[static_cast<int>(SpeciesID::Mt)] = std::move(speciesAttribute("Mt"));
	speciesTable[static_cast<int>(SpeciesID::Ds)] = std::move(speciesAttribute("Ds"));
	speciesTable[static_cast<int>(SpeciesID::Rg)] = std::move(speciesAttribute("Rg"));
	speciesTable[static_cast<int>(SpeciesID::Cn)] = std::move(speciesAttribute("Cn"));
	speciesTable[static_cast<int>(SpeciesID::Uut)] = std::move(speciesAttribute("Uut"));
	speciesTable[static_cast<int>(SpeciesID::Fl)] = std::move(speciesAttribute("Fl"));
	speciesTable[static_cast<int>(SpeciesID::Uup)] = std::move(speciesAttribute("Uup"));
	speciesTable[static_cast<int>(SpeciesID::Lv)] = std::move(speciesAttribute("Lv"));
	speciesTable[static_cast<int>(SpeciesID::Uus)] = std::move(speciesAttribute("Uus"));
	speciesTable[static_cast<int>(SpeciesID::Uuo)] = std::move(speciesAttribute("Uuo"));
	speciesTable[static_cast<int>(SpeciesID::user01)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user02)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user03)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user04)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user05)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user06)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user07)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user08)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user09)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user10)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user11)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user12)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user13)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user14)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user15)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user16)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user17)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user18)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user19)] = std::move(speciesAttribute());
	speciesTable[static_cast<int>(SpeciesID::user20)] = std::move(speciesAttribute());
}

species::~species() {}

inline int species::getNumberOfSpecies() { return static_cast<int>(speciesTable.size()); }

inline std::string species::getSpeciesName(int const index)
{
	if (index < getNumberOfSpecies() && index > -1)
	{
		return speciesTable[index].name;
	}
	return "unknown";
}

inline std::string species::getSpeciesName(SpeciesID const index) { return getSpeciesName(static_cast<int>(index)); }

int species::getSpeciesID(std::string const &SpeciesName)
{
	umuq::parser p;
	auto speciesName = p.tolower(SpeciesName);

	if (speciesName.size() < 4 || speciesName.substr(0, 4) != "user")
	{
		speciesName = p.toupper(speciesName, 0, 1);
	}

	auto search = speciesMap.find(speciesName);
	if (search != speciesMap.end())
	{
		return static_cast<int>(search->second);
	}

	UMUQWARNING("Species : ", SpeciesName, " does not exist in the database!");
	return std::numeric_limits<int>::min();
}

inline speciesAttribute species::getSpecies(int const index)
{
	if (index < getNumberOfSpecies() && index > -1)
	{
		return speciesTable[index];
	}
	return speciesAttribute();
}

inline speciesAttribute species::getSpecies(SpeciesID const index)
{
	return getSpecies(static_cast<int>(index));
}

speciesAttribute species::getSpecies(std::string const &SpeciesName)
{
	return getSpecies(getSpeciesID(SpeciesName));
}

} // namespace umuq

#endif // UMUQ_SPECIESNAME
