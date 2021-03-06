#ifndef UMUQ_FACTORIAL_H
#define UMUQ_FACTORIAL_H

#include "core/core.hpp"

#include <cmath>

#include <limits>

namespace umuq
{

/*!
 * \file numerics/factorial.hpp
 *
 */

/*! \class maxFactorial
 * \ingroup Numerics_Module
 *
 * \brief Predefined max factorial
 *
 * \tparam DataType data type one of float, double, int
 */
template <class DataType>
struct maxFactorial
{
    static unsigned int const value = 0;
};

/*! \class maxFactorial
 * \ingroup Numerics_Module
 *
 * \brief Predefined max factorial (specialized for float)
 */
template <>
struct maxFactorial<float>
{
    static unsigned int const value = 34;
};

/*! \class maxFactorial
 * \ingroup Numerics_Module
 *
 * \brief Predefined max factorial (specialized for double)
 */
template <>
struct maxFactorial<double>
{
    static unsigned int const value = 170;
};

/*! \class maxFactorial
 * \ingroup Numerics_Module
 *
 * \brief Predefined max factorial (specialized for int)
 */
template <>
struct maxFactorial<int>
{
    static unsigned int const value = 11;
};

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial
 *
 * \tparam DataType Data type one of float, double, long double
 *
 * \param n  Input number
 *
 * \returns The factorial of n for type float, double, long double, int, unsigned int, long int and Error for anything else
 */
template <class DataType>
inline DataType uncheckedFactorial(unsigned int n)
{
    UMUQFAIL("The uncheckedFactorial is not implemented for this type!");
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for float)
 *
 * \param n  Input number
 *
 * \returns The float type factorial of n
 */
template <>
inline float uncheckedFactorial<float>(unsigned int const n)
{
    static float const factorials[] =
        {
            1.0f,
            1.0f,
            2.0f,
            6.0f,
            24.0f,
            120.0f,
            720.0f,
            5040.0f,
            40320.0f,
            362880.0f,
            3628800.0f,
            39916800.0f,
            479001600.0f,
            6227020800.0f,
            87178291200.0f,
            1307674368000.0f,
            20922789888000.0f,
            355687428096000.0f,
            6402373705728000.0f,
            121645100408832000.0f,
            0.243290200817664e19f,
            0.5109094217170944e20f,
            0.112400072777760768e22f,
            0.2585201673888497664e23f,
            0.62044840173323943936e24f,
            0.15511210043330985984e26f,
            0.403291461126605635584e27f,
            0.10888869450418352160768e29f,
            0.304888344611713860501504e30f,
            0.8841761993739701954543616e31f,
            0.26525285981219105863630848e33f,
            0.822283865417792281772556288e34f,
            0.26313083693369353016721801216e36f,
            0.868331761881188649551819440128e37f,
            0.29523279903960414084761860964352e39f};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for long double)
 *
 * \param n  Input number
 *
 * \returns The long double type factorial of n
 */
template <>
inline long double uncheckedFactorial<long double>(unsigned int const n)
{
    static long double const factorials[] =
        {
            1l,
            1l,
            2l,
            6l,
            24l,
            120l,
            720l,
            5040l,
            40320l,
            362880.0l,
            3628800.0l,
            39916800.0l,
            479001600.0l,
            6227020800.0l,
            87178291200.0l,
            1307674368000.0l,
            20922789888000.0l,
            355687428096000.0l,
            6402373705728000.0l,
            121645100408832000.0l,
            0.243290200817664e19l,
            0.5109094217170944e20l,
            0.112400072777760768e22l,
            0.2585201673888497664e23l,
            0.62044840173323943936e24l,
            0.15511210043330985984e26l,
            0.403291461126605635584e27l,
            0.10888869450418352160768e29l,
            0.304888344611713860501504e30l,
            0.8841761993739701954543616e31l,
            0.26525285981219105863630848e33l,
            0.822283865417792281772556288e34l,
            0.26313083693369353016721801216e36l,
            0.868331761881188649551819440128e37l,
            0.29523279903960414084761860964352e39l,
            0.103331479663861449296666513375232e41l,
            0.3719933267899012174679994481508352e42l,
            0.137637530912263450463159795815809024e44l,
            0.5230226174666011117600072241000742912e45l,
            0.203978820811974433586402817399028973568e47l,
            0.815915283247897734345611269596115894272e48l,
            0.3345252661316380710817006205344075166515e50l,
            0.1405006117752879898543142606244511569936e52l,
            0.6041526306337383563735513206851399750726e53l,
            0.265827157478844876804362581101461589032e55l,
            0.1196222208654801945619631614956577150644e57l,
            0.5502622159812088949850305428800254892962e58l,
            0.2586232415111681806429643551536119799692e60l,
            0.1241391559253607267086228904737337503852e62l,
            0.6082818640342675608722521633212953768876e63l,
            0.3041409320171337804361260816606476884438e65l,
            0.1551118753287382280224243016469303211063e67l,
            0.8065817517094387857166063685640376697529e68l,
            0.427488328406002556429801375338939964969e70l,
            0.2308436973392413804720927426830275810833e72l,
            0.1269640335365827592596510084756651695958e74l,
            0.7109985878048634518540456474637249497365e75l,
            0.4052691950487721675568060190543232213498e77l,
            0.2350561331282878571829474910515074683829e79l,
            0.1386831185456898357379390197203894063459e81l,
            0.8320987112741390144276341183223364380754e82l,
            0.507580213877224798800856812176625227226e84l,
            0.3146997326038793752565312235495076408801e86l,
            0.1982608315404440064116146708361898137545e88l,
            0.1268869321858841641034333893351614808029e90l,
            0.8247650592082470666723170306785496252186e91l,
            0.5443449390774430640037292402478427526443e93l,
            0.3647111091818868528824985909660546442717e95l,
            0.2480035542436830599600990418569171581047e97l,
            0.1711224524281413113724683388812728390923e99l,
            0.1197857166996989179607278372168909873646e101l,
            0.8504785885678623175211676442399260102886e102l,
            0.6123445837688608686152407038527467274078e104l,
            0.4470115461512684340891257138125051110077e106l,
            0.3307885441519386412259530282212537821457e108l,
            0.2480914081139539809194647711659403366093e110l,
            0.188549470166605025498793226086114655823e112l,
            0.1451830920282858696340707840863082849837e114l,
            0.1132428117820629783145752115873204622873e116l,
            0.8946182130782975286851441715398316520698e117l,
            0.7156945704626380229481153372318653216558e119l,
            0.5797126020747367985879734231578109105412e121l,
            0.4753643337012841748421382069894049466438e123l,
            0.3945523969720658651189747118012061057144e125l,
            0.3314240134565353266999387579130131288001e127l,
            0.2817104114380550276949479442260611594801e129l,
            0.2422709538367273238176552320344125971528e131l,
            0.210775729837952771721360051869938959523e133l,
            0.1854826422573984391147968456455462843802e135l,
            0.1650795516090846108121691926245361930984e137l,
            0.1485715964481761497309522733620825737886e139l,
            0.1352001527678402962551665687594951421476e141l,
            0.1243841405464130725547532432587355307758e143l,
            0.1156772507081641574759205162306240436215e145l,
            0.1087366156656743080273652852567866010042e147l,
            0.103299784882390592625997020993947270954e149l,
            0.9916779348709496892095714015418938011582e150l,
            0.9619275968248211985332842594956369871234e152l,
            0.942689044888324774562618574305724247381e154l,
            0.9332621544394415268169923885626670049072e156l,
            0.9332621544394415268169923885626670049072e158l,
            0.9425947759838359420851623124482936749562e160l,
            0.9614466715035126609268655586972595484554e162l,
            0.990290071648618040754671525458177334909e164l,
            0.1029901674514562762384858386476504428305e167l,
            0.1081396758240290900504101305800329649721e169l,
            0.1146280563734708354534347384148349428704e171l,
            0.1226520203196137939351751701038733888713e173l,
            0.132464181945182897449989183712183259981e175l,
            0.1443859583202493582204882102462797533793e177l,
            0.1588245541522742940425370312709077287172e179l,
            0.1762952551090244663872161047107075788761e181l,
            0.1974506857221074023536820372759924883413e183l,
            0.2231192748659813646596607021218715118256e185l,
            0.2543559733472187557120132004189335234812e187l,
            0.2925093693493015690688151804817735520034e189l,
            0.339310868445189820119825609358857320324e191l,
            0.396993716080872089540195962949863064779e193l,
            0.4684525849754290656574312362808384164393e195l,
            0.5574585761207605881323431711741977155627e197l,
            0.6689502913449127057588118054090372586753e199l,
            0.8094298525273443739681622845449350829971e201l,
            0.9875044200833601362411579871448208012564e203l,
            0.1214630436702532967576624324188129585545e206l,
            0.1506141741511140879795014161993280686076e208l,
            0.1882677176888926099743767702491600857595e210l,
            0.237217324288004688567714730513941708057e212l,
            0.3012660018457659544809977077527059692324e214l,
            0.3856204823625804217356770659234636406175e216l,
            0.4974504222477287440390234150412680963966e218l,
            0.6466855489220473672507304395536485253155e220l,
            0.8471580690878820510984568758152795681634e222l,
            0.1118248651196004307449963076076169029976e225l,
            0.1487270706090685728908450891181304809868e227l,
            0.1992942746161518876737324194182948445223e229l,
            0.269047270731805048359538766214698040105e231l,
            0.3659042881952548657689727220519893345429e233l,
            0.5012888748274991661034926292112253883237e235l,
            0.6917786472619488492228198283114910358867e237l,
            0.9615723196941089004197195613529725398826e239l,
            0.1346201247571752460587607385894161555836e242l,
            0.1898143759076170969428526414110767793728e244l,
            0.2695364137888162776588507508037290267094e246l,
            0.3854370717180072770521565736493325081944e248l,
            0.5550293832739304789551054660550388118e250l,
            0.80479260574719919448490292577980627711e252l,
            0.1174997204390910823947958271638517164581e255l,
            0.1727245890454638911203498659308620231933e257l,
            0.2556323917872865588581178015776757943262e259l,
            0.380892263763056972698595524350736933546e261l,
            0.571338395644585459047893286526105400319e263l,
            0.8627209774233240431623188626544191544816e265l,
            0.1311335885683452545606724671234717114812e268l,
            0.2006343905095682394778288746989117185662e270l,
            0.308976961384735088795856467036324046592e272l,
            0.4789142901463393876335775239063022722176e274l,
            0.7471062926282894447083809372938315446595e276l,
            0.1172956879426414428192158071551315525115e279l,
            0.1853271869493734796543609753051078529682e281l,
            0.2946702272495038326504339507351214862195e283l,
            0.4714723635992061322406943211761943779512e285l,
            0.7590705053947218729075178570936729485014e287l,
            0.1229694218739449434110178928491750176572e290l,
            0.2004401576545302577599591653441552787813e292l,
            0.3287218585534296227263330311644146572013e294l,
            0.5423910666131588774984495014212841843822e296l,
            0.9003691705778437366474261723593317460744e298l,
            0.1503616514864999040201201707840084015944e301l,
            0.2526075744973198387538018869171341146786e303l,
            0.4269068009004705274939251888899566538069e305l,
            0.7257415615307998967396728211129263114717e307l};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for double)
 *
 * \param n  Input number
 *
 * \returns The double type factorial of n
 */
template <>
inline double uncheckedFactorial<double>(unsigned int const n)
{
    return static_cast<double>(uncheckedFactorial<long double>(n));
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for int)
 *
 * \param n  Input number
 *
 * \returns The int type factorial of n
 */
template <>
inline int uncheckedFactorial<int>(unsigned int const n)
{
    static int const factorials[] =
        {
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            1065353216};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for unsigned int)
 *
 * \param n  Input number
 *
 * \returns The unsigned int type factorial of n
 */
template <>
inline unsigned int uncheckedFactorial<unsigned int>(unsigned int const n)
{
    static unsigned int const factorials[] =
        {
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            1065353216};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for long int)
 *
 * \param n  Input number
 *
 * \returns The long int type factorial of n
 */
template <>
inline long int uncheckedFactorial<long int>(unsigned int const n)
{
    static long int const factorials[] =
        {
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            1065353216};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Predefined unchecked factorial (specialized for long unsigned int)
 *
 * \param n  Input number
 *
 * \returns The long unsigned int type factorial of n
 */
template <>
inline long unsigned int uncheckedFactorial<long unsigned int>(unsigned int const n)
{
    static long unsigned int const factorials[] =
        {
            1,
            1,
            2,
            6,
            24,
            120,
            720,
            5040,
            40320,
            362880,
            3628800,
            1065353216};
    return factorials[n];
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$
 *
 * \tparam DataType data type one of float, double, int, unsigned int, long int, and long unsigned int
 *
 * \param n Input number
 *
 * \returns The factorial of n : \f$\left(n!\right)\f$ for any types of float, double, int, unsigned int, long int, long unsigned int and Error for anything else
 */
template <class DataType>
inline DataType factorial(unsigned int const n)
{
    UMUQFAIL("Factorial is not implemented for this type!");
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for float)
 *
 * \param n Input number
 *
 * \returns The float type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline float factorial(unsigned int const i)
{
    if (i <= maxFactorial<float>::value)
    {
        return uncheckedFactorial<float>(i);
    }

    double const result = std::tgamma(static_cast<double>(i + 1));

    if (result > std::numeric_limits<float>::max())
    {
        UMUQWARNING("Overflowed value!");
        return static_cast<float>(result);
    }

    return static_cast<float>(std::floor(result + 0.5f));
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for double)
 *
 * \param n Input number
 *
 * \returns The double type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline double factorial(unsigned int const i)
{
    if (i <= maxFactorial<double>::value)
    {
        return uncheckedFactorial<double>(i);
    }

    long double const result = std::tgamma(static_cast<long double>(i + 1));

    if (result > std::numeric_limits<double>::max())
    {
        UMUQWARNING("Overflowed value!");
        return static_cast<double>(result);
    }

    return static_cast<double>(std::floor(result + 0.5));
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for long unsigned int)
 *
 * \param n Input number
 *
 * \returns The long unsigned int type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline long unsigned int factorial(unsigned int const i)
{
    return (i <= maxFactorial<int>::value ? uncheckedFactorial<long unsigned int>(i) : static_cast<long unsigned int>(i) * factorial<long unsigned int>(i - 1));
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for int)
 *
 * \param n Input number
 *
 * \returns The int type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline int factorial(unsigned int const i)
{
    if (i <= maxFactorial<int>::value)
    {
        return uncheckedFactorial<int>(i);
    }

    long unsigned int const result = i * factorial<long unsigned int>(i - 1);

    if (result > std::numeric_limits<int>::max())
    {
        UMUQWARNING("Overflowed value!");
    }
    return static_cast<int>(result);
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for unsigned int)
 *
 * \param n Input number
 *
 * \returns The unsigned int type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline unsigned int factorial(unsigned int const i)
{
    if (i <= maxFactorial<int>::value)
    {
        return uncheckedFactorial<unsigned int>(i);
    }

    long unsigned int const result = i * factorial<long unsigned int>(i - 1);

    if (result > std::numeric_limits<unsigned int>::max())
    {
        UMUQWARNING("Overflowed value!");
    }
    return static_cast<unsigned int>(result);
}

/*!
 * \ingroup Numerics_Module
 *
 * \brief Compute the factorial of n : \f$\left(n!\right)\f$ (specialized for long int)
 *
 * \param n Input number
 *
 * \returns The long int type factorial of n : \f$\left(n!\right)\f$
 */
template <>
inline long int factorial(unsigned int const i)
{
    if (i <= maxFactorial<int>::value)
    {
        return uncheckedFactorial<long int>(i);
    }

    long unsigned int const result = i * factorial<long unsigned int>(i - 1);

    if (result > std::numeric_limits<long int>::max())
    {
        UMUQWARNING("Overflowed value!");
    }
    return static_cast<long int>(result);
}

} // namespace umuq

#endif // UMUQ_FACTORIAL
