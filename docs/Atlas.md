

<!-- Start of picture text -->
FiO SAN |<br>ZO in go<br>yy | |_Ke Peet?<br><!-- End of picture text -->

**ATLAS: Adaptive Multi-Scale Framework for Structural Shape Optimization Using Extended Physics-Informed Neural Networks** 

A Title Proposal 

Presented to the Faculty of the Senior High School Department Davao City National High School 

In Partial Fulfillment of the Requirements for the Subject 

**Research 1** 

Submitted by: 

**Alag, Clefford John P. Dio, Yuan Carlo R. Haguriles, Rey Mark E. Laganson, Xavier Troy G. Lao, Zepto Dyne F. Salcedo, Marcoux Gaddiel V.** Grade 12 — STEM NEWTON 

Submitted to: 

**Christian S. Englisa** Research Adviser July 2026 

_Title Proposal: ATLAS_ 

## **ABSTRACT** 

Finite Element Method (FEM) simulations are widely used in structural engineering, as analysis and design tools for load bearing systems. Although FEM is accurate, it is also time consuming and can get very expensive if thousands of simulations are needed in an iterative design process. In the field of scientific machine learning, Physics-Informed Neural Networks (PINNs) have recently come to the fore as a class of models that embed governing physical laws for the training of the neural network. But typical PINNs become difficult to apply to complex real-world geometries, multi-scale stress concentrations and inverse design tasks. This study is proposing ATLAS, an Adaptive Multi-Scale Framework for Structural Design based on Extended Physics-Informed Neural Networks (XPINNs), a domain decomposition-based deep learning framework for solving partial differential equations that describe structural mechanics problems. ATLAS breaks down the complex structural domains into smaller subdomains, learns a unique neural network for each subdomain, and enforces the continuity conditions on the interfaces to generate high fidelity stress, strain and displacement predictions across the structure. The framework also includes an inverse design module for shape optimization using automatic differentiation, with no need for large labeled datasets. Validation is conducted using FEM results for canonical 2D structural problems such as a cantilever beam, L-shaped bracket and plate with hole. The end goal of ATLAS is to prove that the multi-scale frameworks based on XPINN can be accurate as FEM and provide gradient-driven structural shape design, which is lacking in all existing single-domain PINN-based structural analysis methods. 

**_Keywords:_** _Physics-Informed Neural Networks (PINNs), Extended Physics-Informed Neural Networks (XPINNs), Structural Design, Shape Optimization, Domain Decomposition, Multi-Scale Analysis, Inverse Design, Finite Element Method (FEM)_ 

2 

_Title Proposal: ATLAS_ 

## **I. INTRODUCTION** 

## **_A. Background of the Study_** 

Structural systems such as bridges, buildings, machine components, and aerospace structures are the supporting infrastructure of today's society, and their performance failures have devastating consequences. The Philippines being one of the world's most disaster-prone countries, the structural integrity of its infrastructure is a matter of life and death. The Philippines is a country that experiences an average of 20 earthquakes per day annually and around 20 tropical cyclones per year enter the Philippine Area of Responsibility (PAR) (Statista, 2024). Total damages in 2024 amounted to more than 43 billion Philippine pesos, with the majority of the damages being caused by destroyed infrastructure (Statista, 2024). The Cebu earthquake that occurred in September 2025 resulted in 79 deaths and 559 injuries, with over 134,000 houses damaged and infrastructure losses valued at ₱6.76 billion (IFRC, 2025). The recent collapse of various bridges has exposed significant weaknesses in the infrastructure system, including poorly designed bridges, poor maintenance, and unchecked deterioration (Orale, 2025). Therefore, the government spent nearly USD 26 billion on infrastructure in 2024 (Asian Insiders, 2025) and the Philippines has a $400 billion infrastructure investment deficit (Indo-Pacific Partnership, 2025). However, investment in infrastructure is still less than that of peers in the region, and logistics costs are also one of the highest in Asia (ADB, 2025). Structural simulation, a computational process of forecasting the response of materials and geometries to loads, earthquakes and environmental forces, is at the heart of designing structures to resist these forces. The Finite Element Method (FEM) is the most popular method for this purpose, but it is costly as shown to be consuming about 50% of the total calculation cost in structural optimization processes (Chandrasekhar & Suresh, 2021). The computing power of traditional FEM is a major drawback 

3 

_Title Proposal: ATLAS_ 

for a nation with the need to design, assess and rebuild infrastructure in unprecedented speed and scale, and there is a pressing need for more speedy, smarter, physics-aware solutions. 

In computational structural engineering, design iteration is still a costly time-consuming process, especially for researchers and engineers working in academic or resource-limited environments. The process of structural optimization usually takes hundreds to thousands of FEM simulations to reach an optimum design, and the cost of the simulation is not linear with respect to the mesh resolution. This is a problem for students and small teams that don't have access to high-performance computing systems or expensive commercial software, as it renders meaningful design exploration of the structure inaccessible. This is a problem for students and small teams who do not have access to high performance computing or expensive commercial software as meaningful structural design exploration is not accessible. 

There are two types of machine learning substitutes. Data-driven surrogates can be very fast to compute an approximation of FEM results, but require huge labeled data sets (often generated by the very FEM simulation they aim to replace). An alternative solution is offered by Physics-Informed Neural Networks (PINNs), which connect governing equations directly to the loss function and require only a few or no labeled data points (Karniadakis et al., 2021). The use of PINN in solid mechanics is still in its early stages, however, compared to the fast advances made by PINN in solving fluid mechanics problems, the application of PINN may sometimes be more difficult for general solid mechanics problems, due to the strong discontinuities and irregularities found in the solution fields (Gu et al., 2025). Moreover, vanilla PINNs have difficulties in capturing multi-scale features and hence suffer from large errors in models and 

4 

_Title Proposal: ATLAS_ 

poor convergence. Though some structural analysis frameworks based on PINNs have demonstrated their potential, they have all the limitations of single-domain architectures, which inherit the geometric limitation and multi-scale limitation. To tackle these challenges, domain decomposition methods that partition the problem domain into non-overlapping subdomains have been proposed, and Hu et al. (2022) showed that extended PINNs (XPINNs) based on this approach greatly surpass single domain PINNs in terms of generalization. There is no work in the literature, however, that has applied XPINNs to structural inverse design which also leaves a gap within the literature. This study introduces an Adaptive Multi-Scale Framework for Structural Design (ATLAS), which is based on Extended Physics-Informed Neural Networks (XPINNs) (Hu et al., 2022). In contrast to the standard PINN that only has one neural network over the entire domain, XPINN breaks the structural geometry into n subdomains, each of which has its own neural network, and ensures displacement and traction continuity at subdomain boundaries. This domain decomposition approach directly tackles the geometric complexity and multi-scale nature that single-domain PINNs are unable to solve. Building on this XPINN forward solver, ATLAS adds an inverse design module that uses automatic differentiation to optimize a structural parameter (such as hole radius or fillet geometry) without performing a single FEM simulation, by minimizing the compliance of the structure. The framework is realized as a prototype implemented in Python and validated on three canonical 2D structural problems with FEM, with the goal of showing that an XPINN-based multi-scale framework can obtain FEM comparable accuracy, but with a lower computational cost, while enabling the gradient-driven design of structural shape. 

5 

_Title Proposal: ATLAS_ 

## **_B. Statement of the Problem_** 

As of today, several structures are proven to be substandard and are causing hazards that can potentially harm many people surrounding the structure. Traditional Neural Networks can take input data, scan and learn the data, and provide answers through the data given. Additionally, traditional Neural Networks take the data slowly which causes slight inefficiency.  This Study aims to design an Extended Physics-Informed Neural Network that can improve its efficiency and take more complex data such as, geometry, loads, supports, and material properties. This design can adapt to multi-scale framework for structural design using Extended Physics-Informed Neural Networks (XPINN). Its main objective is to help people deal with substandard structures, give simulations and results, and give answers regarding what can be improved or be fixed on a specific structure to avoid any potential harm to people surrounding the structure. Specifically, this study seeks to answer the following questions: 

1. How can the Extended Physics-Informed Neural Networks (XPINN) be designed and implemented to process complex datasets more efficiently than a neural network? 

2. What is the prediction accuracy of the Extended Physics-Informed Neural Networks (XPINN) compared to a regular neural network when tested on complex datasets? 

3. How does the Extended Physics-Informed Neural Networks (XPINN) perform in terms of processing time and computational efficiency compared to a regular neural network? 

4. How do users evaluate the Extended Physics-Informed Neural Networks (XPINN) in terms of usability, reliability, and effectiveness? 

5. What is the difference in computational resource usage (e.g., memory and processing power) between the Extended Physics-Informed Neural Networks (XPINN) and a traditional neural network? 

6 

_Title Proposal: ATLAS_ 

## **II. OBJECTIVES OF THE STUDY** 

## **_A. General Objective_** 

This study aims to develop ATLAS (Adaptive Multi-Scale Framework for Structural Design), an XPINN-based physics-informed learning system capable of performing forward structural 

analysis and inverse structural design on complex 2D geometries with minimal data requirements, validated against Finite Element Method benchmarks. 

## **_B. Specific Objectives_** 

Specifically, this study aims to: 

1. Design the ATLAS framework architecture: XPINN subdomain decomposition strategy, enforcement of interface continuity and the inverse design module with automatic differentiation. 

2. To create a Python prototype of ATLAS consisted of: an input module (geometry, material properties, load cases and boundary conditions); an XPINN solver module; an inverse design module; an FEM validation module; and a visualization module. 

3. To assess the accuracy of ATLAS for forward solving in canonical 2D structures (cantilever beam, L-shaped bracket, plate with a hole) by computing relative errors between the displacement and stress fields computed by ATLAS and FEM, with the L2 norm. 

4. To compare the convergence behavior and prediction accuracy of ATLAS (XPINN multi-scale) and a single domain PINN baseline to gauge the specific benefit of domain decomposition for complex and multi-scale structural problems. 

7 

_Title Proposal: ATLAS_ 

5. To determine the effectiveness of the ATLAS inverse shape design module in performing parametric shape optimization (hole geometry, fillet geometry), comparing resulting structural designs against FEM-based parametric optimization in terms of compliance, material usage, and design quality. 

## **III. SIGNIFICANCE OF THE STUDY** 

This study is part of the growing overlap between the field of scientific machine learning and structural engineering, and can benefit several fields and stakeholders. Structural and Mechanical Engineers will find a framework with reduced computational cost to ease the burden of design iteration. ATLAS provides a possibility of gradient-driven structural design, where the use of hundreds of FEM simulations is not necessary, and therefore fast design exploration is possible even in resource constrained environments. A framework specifically designed for geometric complexity and multi-scale stress features that are not captured by standard PINNs would be useful to Civil/ Aerospace Engineers for solving problems with complex load-bearing geometries, including brackets, trusses, wing ribs and bridge components. Scientific Machine Learning researchers will have access to such a benchmark study of the application of XPINN domain decomposition to the field of solid mechanics and structural shape design, which has not been explored in the literature before. 

With numerous applications such as computational mechanics, structural simulation, shape optimization, etc., structural engineering is a broad domain and ATLAS offers a unified differentiable framework that encompasses all three. The open source Python prototype will 

8 

_Title Proposal: ATLAS_ 

allow for the application of physics-informed learning to practical structural problems; this prototype will lower the barrier to entry for PINN-based research in mechanics for students and educators in engineering and computational science. Future researchers in PINN-based structural optimization can benefit from the ATLAS framework as a basis for implementation of topology optimization, 3D, nonlinear material and multi-physics structural design. 

## **IV. SCOPE AND LIMITATIONS** 

## **_A. Scope_** 

Create 2D XPINN based structural analysis and inverse shape design system (ATLAS) in python with PyTorch. Consider plane stress problem linear elasticity characterized by Navier-Cauchy equilibrium equations and Hooke's constitutive law. Solve three benchmark problems using ATLAS: (1) 2D cantilever beam, (2) L-shaped bracket with stress concentration, (3) plate with circular hole under tension. Use domain decomposition with 2-6 XPINN subdomains per benchmark, enforcing continuity of interfaces between subdomains. Use the inverse design module with automatic differentiation to calculate the shape optimization (parametric hole/fillet geometry). Test all forward-problem results with the FEM solutions obtained with FEniCS or another open source FEM solver. Carry out comparisons of the inverse shape design outputs with FEM based parametric optimisation results. Create field visualisations (displacement, stress, von Mises stress) for all test cases. 

9 

_Title Proposal: ATLAS_ 

## **_B. Limitations_** 

- Address three-dimensional (3D) structural problems; the framework is scoped to 2D 

plane stress. 

- Handle nonlinear material behavior (plasticity, hyperelasticity) or geometric nonlinearity (large deformations). 

- Address dynamic loading, time-dependent structural response, or fatigue analysis. 

- Include experimental validation; all validation is computational (FEM vs. XPINN). 

- Perform topology optimization (density field-based material layout); this is identified as future work. 

- Target deployment-ready or real-time industrial software; ATLAS is a research prototype. 

- Address multi-material or anisotropic material configurations in this phase. 

- Guarantee global optimality of inverse design solutions; the framework is gradient-based 

and subject to local minima. 

10 

_Title Proposal: ATLAS_ 

## **V. REVIEW OF RELATED LITERATURE** 

## **_A. Total value of government expenditures on infrastructure and other capital outlays in the Philippines from 2014 to 2024_** 

The Philippines is one of the worst hit countries in Southeast Asia in terms of structural engineering, being situated at the Pacific Ring of Fire and in the typhoon belt. In 2023, the total damages to infrastructures due to major natural extreme events and disasters in the Philippines reached about PHP13.4 billion, an improvement from the previous year. Total damages in the Philippines from major extreme natural events and disasters were estimated to be around 23 billion Philippine pesos in the same year. These regular losses are not only due to the natural forces at play but also due to a lack of strategies in the design, analysis and optimization of structures to withstand extreme loading. The development of infrastructure is critical for any country, as it directly contributes to the economy's growth and has a positive impact on the people's lives; however, in the Philippines, infrastructure development has been difficult because of the many persistent problems. To overcome this, the Philippine Government has greatly increased structural investments. In 2024, the Philippine government allocated about 1.3 trillion Philippine pesos (P1.3 trillion) for infrastructure and capital outlays, an increase of 10 percent from the previous year, and mostly allocated to various flagship projects within Build-Better-More. Nevertheless, these vast investments have not resulted in the development of any computational tools for designing and verifying these structures that are competing with the traditional and reliable Finite Element Method (FEM) computation, which is extremely time consuming and expensive for engineers and institutions especially under resource-limited environments. Structural simulation is the computational foundation of contemporary engineering design. Structural engineering is a wide-ranging discipline with many practical applications such as computational mechanics, structural simulation and topology optimization. 

11 

_Title Proposal: ATLAS_ 

The need for faster and smarter computational tools to design structurally optimal systems under complex loading conditions is growing, particularly in countries that are subjected to hazards such as the Philippines. The need to meet the requirements of the modern design workflow is the problem that this study tries to solve, and the gap between what traditional FEM can provide and what it requires is just that. 

## **_B. Existing Approaches to Structural Simulation and Their Limitations_** 

Traditionally, structural analysis is achieved by FEM-based simulation, where a structure 

is divided into thousands of finite elements and the governing equations are numerically solved. FEM is a reliable tool, but has fundamental limitations in the iterative design process. 

FEA-based topology optimization methods have several limitations, such as solving nonlinear problems which are often complex, and design difficulties regarding dynamic loading and fracture. Furthermore, the maximum error of PINNs is much larger than that of FEM, and the computation time of PINNs is also much longer than that of FEM, as tens of thousands of epochs are required for training. It points out an important trade-off: Flexibility of early PINN implementations comes at a price of convergence challenges against classical solvers, while accuracy of FEM comes at a price of computational rigidity. Purely data-driven machine learning surrogates have been presented as alternatives to the use of FEM. However, such methods are dependent on huge numbers of labeled data which are usually only generated by the very FEM computations they aim to replace. 

While promising for other areas of mechanics, physics-informed neural networks (PINNs) that incorporate governing equations, symmetries, and physical constraints into the architecture of machine learning (ML) are relatively unexplored for structural design, and have the potential to enhance generalization and lower the amount of data needed. A more principled alternative is 

12 

_Title Proposal: ATLAS_ 

known as Physics-Informed Neural Networks (PINNs). Research has shown that PINNs are able to achieve higher accuracy, stability, and computational efficiency compared to conventional artificial neural networks and analytical solutions. But, it has been well documented that single domain PINNs are deficient in structural mechanics. While the vanilla PINN holds promise for forward problems, its applications in surrogate modeling in the field of solid mechanics are still under-explored. More critically, In the few topology optimization studies that have been conducted using PINN, mostly high-performance designs were obtained compared to the traditional methods, but relatively very-high computational times were the result. To fill this gap, several topology optimization frameworks based on PINN have been proposed. Optimal topologies can be obtained by CPINNTO with neither labeled data nor FEA, and optimal structures can be obtained for various applications such as compliance minimization problems, multi-constrained problems and 3D problems, with structures that have good compliances close to those obtained by density based methods. Similarly, The continuous formulation of topology optimization, coupled with the continuous adjoint method, and a variant of PINN, the Deep Energy Method, completely replaces traditional PDE solution procedures like FEM for a PINN-based topology optimization method without FEM. All of these structures, however, are based on the single domain PINN approach, which suffers from geometric and multi-scale limitations that do not allow it to accurately represent complex real-world structural geometries. 

13 

_Title Proposal: ATLAS_ 

# **_C. Extended Physics-Informed Neural Networks (XPINNs) and Domain Decomposition as the Core Technology_** 

Single-domain PINNs were found to have some drawbacks and domain decomposition strategies were added to the PINN framework. XPINNs divide the physical domain, and allocate the sub-networks to the local regions based on interface continuity conditions (Hu et al., 2022). Hu et al. (2022) illustrated that, with multi-scale features and irregular boundaries, extended physics-informed neural networks can achieve statistically significant accuracy gains over single domain PINNs, and they can improve generalization. The domain decomposition approach is readily applicable to structural mechanics as it permits a higher resolution to be achieved in the vicinity of stress concentrations, discontinuities and material interfaces, precisely where the multi-scale features that make single-domain PINNs problematic are present. Recent developments have continued to reinforce the use of the domain decomposition version of PINNs in solid mechanics. Shukla et al. (2021): Shukla et al. proposed a parallel physics-informed neural networks (PINN) approach by domain decomposition that shows that subdomain-based training leads to better accuracy and training efficiency on complex problems. Introducing the domain decomposition strategy, PINNs with multi-network architectures have been successfully used to tackle multi-material problems in solid mechanics, soil moisture dynamics and pile-soil interaction, all of which have important implications for multi-scale challenges. The ability of physics-informed networks to be used to solve inverse problems in engineering structures under various loading conditions has additionally been showcased in a transfer learning context for the solution of inverse problems in engineering structures under different loading conditions (Xu et al., 2023). This shows that the XPINN architecture can be expanded to design optimization tasks. This study takes a decision to use XPINN as the core technology for ATLAS, because XPINN satisfies all three key challenges highlighted in the 

14 

_Title Proposal: ATLAS_ 

literature: (1) geometric complexity: spatial domain decomposition that adapts to irregular boundaries; (2) multi-scale stress features: subdomain-level resolution control; and (3) inverse shape design: automatic differentiation of the entire differentiable domain decomposition. ATLAS is the first to use the XPINN domain decomposition with the structural shape optimization. 

## **_D. Research Gap_** 

Although great advances have been made in both XPINN domain decomposition and PINN-based structural analysis, there is no existing research that integrates the two. Existing PINN-based structural analysis tools such as physics-informed deep learning framework for solid mechanics (Haghighat et al., 2021) and transfer learning-based PINNs for engineering structures (Xu et al., 2023) show great promise for forward solving accuracy but are solely based on one domain architectures, thus suffer from geometric and multi-scale limitations when being applied to complex real-world structural geometries. In the domain decomposition direction, Hu et al. (2022) demonstrated that XPINNs outperform single-domain PINNs in terms of generalization performance and Shukla et al. (2021) reported that parallel PINNs using domain decomposition algorithms are more accurate and faster to train — but both works did not use XPINN for structural inverse design. More recently, Gu et al. (2025) specifically pointed out that PINNs have not been applied significantly to solid mechanics problems and that complicated geometry structures still have fundamental challenges to be addressed. ATLAS fills this exact need by introducing XPINN subdomain decomposition into the forward solver of a structural inverse shape design framework, merging multi-scale resolution control with automatic differentiation-based shape design into a single differentiable pipeline. There is no previous 

15 

_Title Proposal: ATLAS_ 

study in the literature that has been able to demonstrate this combination (XPINN domain decomposition for structural shape optimization) 

## **VI. PRELIMINARY METHODOLOGY** 

## **_A. Research Design_** 

This study follows a Developmental Research Design, involving the systematic design, development, testing, and evaluation of a computational prototype. The study proceeds in three phases: (1) forward solver development and validation, (2) multi-scale benchmark testing, and 

(3) shape optimization implementation and evaluation. Each phase builds on the results of the previous, with iterative refinement guided by FEM validation metrics. 

## **_B. System Architecture / Conceptual Framework_** 



<!-- Start of picture text -->
r<br><!-- End of picture text -->

16 

_Title Proposal: ATLAS_ 

Mathematical Foundation. The ATLAS framework is based on the Navier–Cauchy equilibrium equations for linear elasticity, which govern the relationship between displacement and external forces in the structural domain. The governing equation is expressed as: 



where u represents the displacement vector, μ and λ are the Lamé constants, and f represents the external force. After determining the displacement field, the framework calculates strain and stress using Hooke's constitutive law: 



where σ is stress, D is the elasticity matrix, and ε is strain. For the plane stress problems considered in this study, the corresponding elasticity matrix is applied. 

The XPINN framework uses domain decomposition to divide the complete structural domain 

into wsmaller subdomains: 



<!-- Start of picture text -->
N<br>2= (JQ;<br>i=1<br><!-- End of picture text -->

17 

_Title Proposal: ATLAS_ 

where Ω represents the complete structural domain and Ωᵢ represents each XPINN subdomain. This decomposition allows the model to process complex structural regions through separate sub-networks while maintaining interface continuity. 

The ATLAS framework is organized as a modular platform in Python with five interconnected components, all of which are required to enable forward structural analysis and inverse structural shape design using physics-informed learning. The system is a sequential pipeline: the input is the structural problem definition and the output of the XPINN solver is given as input to the inverse design module and the validation module; all the results are given by the visualization module. The geometry is supplied as Signed Distance Functions (SDFs) or by analytical definition of boundaries, material properties (Young's modulus E and Poisson's ratio ν), load conditions (point loads, distributed loads, and body forces), and support and boundary conditions (Dirichlet and Neumann conditions). The XPINN Solver Module consists of the core computational engine, which divides the structural geometry into 2 to 6 non-overlapping subdomains, each having its own fully-connected neural network with 4 to 6 hidden layers, 32 to 64 neurons per layer and hyperbolic-tangent activation functions. It includes three types of losses, PDE residual loss, boundary condition loss, and interface continuity loss, as defined by the Navier-Cauchy equilibrium equations, prescribed displacements and tractions, and displacement and traction continuity across subdomain boundaries, respectively. The optimizer for training is Adam, and for fine-tuning, L-BFGS. The Inverse Shape Design Module optimizes shapes parametrically by specifying design variables (hole radius, fillet radius or beam height, etc.) and using automatic differentiation through the XPINN solver to compute gradients of the compliance objective, thus letting the shape be iterated towards the optimal solution without 

18 

_Title Proposal: ATLAS_ 

requiring FEM during optimization, although using FEM for validation. All of the ATLAS outputs are compared to FEM reference solutions computed by FEniCS and the Validation Module calculates L2 relative errors in both the displacement field and von Mises stress, compliance difference, and the maximum pointwise error at stress concentration locations. The Visualization Module is used to generate displacement and stress field plots, evolution of shape optimizations, convergence curves and comparison of ATLAS and FEM results side by side created using Matplotlib and Plotly. 

## **_C. Testing and Evaluation Plan_** 

The evaluation of the ATLAS framework was carried out by solving a series of three increasingly complex benchmark problems, each testing the capability of the system to solve a specific problem. The first test problem is a 2D cantilever beam under a tip load, which has an analytical solution as a sanity check and to ensure that the XPINN forward solver correctly calculates the displacement and stress field — this test problem was used for this purpose. The second benchmark is an L-shaped bracket with a re-entrant corner, which adds geometric complexity and a sharp stress concentration to the problem, both of which are known to pose challenges to single domain PINNs, but are tested in the multi-scale domain decomposition of ATLAS. The 3rd benchmark is a plate with a hole, under uniaxial tension, which will be tested for forward accuracy in the vicinity of the hole as well as for shape optimization, allowing ATLAS to optimize the hole radius to minimize compliance. For each benchmark, five independent runs with different random initializations will be executed to check the consistency and repeatability of the results. The four quantitative metrics will be used to assess accuracy. The main figure of merit is the L² norm of the relative error between the displacement field and high-resolution FEMS approximations created using FEniCS, where the goal is to obtain an error 

19 

_Title Proposal: ATLAS_ 

lower than 5% for all forward problems. The second metric is the relative error (L²) of the von Mises stress field, which is significant at locations with stress concentrations, such as the re-entrant corner of an L-bracket and the boundary of the hole in the plate. The third is the compliance difference between ATLAS shape-optimized designs and reference designs derived from parametric design using FEM – with the aim of keeping the compliance of ATLAS designs within 10% of the FEM parametric design. The fourth metric is a convergence comparison between the multi-domain XPINN solver and a single domain vanilla PINN baseline, trained on the same problems. 

## **VII. PRELIMINARY MATERIALS AND BUDGET** 

|**Component / Material**|**Qty**|**Unit Cost**|**Total Cost**|**Source / Supplier**|
|---|---|---|---|---|
|Python|1|₱0|₱0|Online (Open-source)|
|PyTorch|1|₱0|₱0|Online (Open-source)|
|FeniCS|1|₱0|₱0|Online (Open-source)|
|Matplotlib|1|₱0|₱0|Online (Open-source)|
|NumPy|1|₱0|₱0|Online (Open-source)|
|Overleaf (LaTeX)|1|₱0|₱0|Online (Open-source)|
||**ESTIMAT**|**ED TOTAL:**|**₱0**||



## **VIII. PROPOSED PROJECT TIMELINE** 

|**Task / Phase**|**Jun**|**Jul**|**Aug**|**Sep**|**Oct**|
|---|---|---|---|---|---|
|Planning (Title Proposal)||||||
|Prototype Design Analysis||||||
|Materials Procurement||||||
|Building/Making Prototype Version 1||||||
|Building/Making Prototype Testing 1||||||



20 

_Title Proposal: ATLAS_ 

|Iteration (Prototype Version 2)|
|---|
|Final Testing|
|Paper Writing (For DSTF Submission)|
|Poster + Defense Prep.|





## **REFERENCES** 

- Asian Development Bank. (2025). _From potential to progress: Future-proofing Philippine growth_ . <u>https://www.adb.org/news/features/potential-progress-future-proofing-philippine-growth</u> 

- Chandrasekhar, A., & Suresh, K. (2021). TOuNN: Topology optimization using neural networks. _Structural and Multidisciplinary Optimization, 63_ (3), 1135–1149. <u>https://doi.org/10.1007/s00158-020-02748-4</u> 

- Department of Budget and Management, Philippines. (2025). _Total value of government expenditures on infrastructure and other capital outlays in the Philippines from 2014 to 2024_ . Statista. <u>https://www.statista.com/statistics/1479466/philippine-government-expenditures-on-infra structure/</u> 

- Gu, B., Wang, Z., & Xie, Y. M. (2025). Finite-PINN: A physics-informed neural network architecture for solving solid mechanics problems with general geometries. _Journal of Computational Physics, 519_ , 113444. https://doi.org/10.1016/j.jcp.2024.113444 

- Haghighat, E., Raissi, M., Moure, A., Gomez, H., & Juanes, R. (2021). A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. _Computer Methods in Applied Mechanics and Engineering, 379_ , 113741. <u>https://doi.org/10.1016/j.cma.2021.113741</u> 

- Hu, Z., Jagtap, A. D., Karniadakis, G. E., & Kawaguchi, K. (2022). When do extended physics-informed neural networks (XPINNs) improve generalization? _SIAM Journal on Scientific Computing, 44_ (5), A3158–A3182. https://doi.org/10.1137/21M1447039 

- Indo-Pacific Partnership for Prosperity. (2025). _Coalition for emerging market infrastructure investment_ . https://www.indopacificpartnership.org/infrastructure-coalition 

- International Federation of Red Cross and Red Crescent Societies. (2025, November 10). _From earthquake to typhoons: Philippines struggles against relentless catastrophes_ . <u>https://www.ifrc.org/press-release/earthquake-typhoons-philippines-struggles-against-rele ntless-catastrophes</u> 

21 

_Title Proposal: ATLAS_ 

- Jackson & Sands Engineering. (2025). _Types of structural failures and common causes_ . <u>https://jacksonandsandsengineering.com/types-of-structural-failures-and-common-causes /</u> 

- Jeong, H., Bai, J., Batuwatta-Gamage, C., Rathnayaka, C., Zhou, Y., & Gu, Y. (2023). A physics-informed neural network-based topology optimization (PINNTO) framework for structural optimization. _Engineering Structures, 278_ , 115484. <u>https://doi.org/10.1016/j.engstruct.2022.115484</u> 

- Jeong, H., Batuwatta-Gamage, C., Bai, J., Xie, Y. M., Rathnayaka, C., Zhou, Y., & Gu, Y. (2023). A complete physics-informed neural network-based framework for structural topology optimization. _Computer Methods in Applied Mechanics and Engineering, 417_ , 116401. <u>https://doi.org/10.1016/j.cma.2023.116401</u> 

- Jeong, H., Batuwatta-Gamage, C., Bai, J., Rathnayaka, C., Zhou, Y., & Gu, Y. (2025). An advanced physics-informed neural network-based framework for nonlinear and complex topology optimization. _Engineering Structures, 322_ , 119194. <u>https://doi.org/10.1016/j.engstruct.2024.119194</u> 

- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. _Nature Reviews Physics, 3_ (6), 422–440. <u>https://doi.org/10.1038/s42254-021-00314-5</u> 

- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F., & Johnson, S. G. (2021). Physics-informed neural networks with hard constraints for inverse design. _SIAM Journal on Scientific Computing, 43_ (6), B1105–B1132. https://doi.org/10.1137/21M1397908 

- MarketsandMarkets. (2025). _Simulation software market report 2025–2030_ . <u>https://www.marketsandmarkets.com/Market-Reports/simulation-software-market-26364 6018.html</u> 

- Orale, R. (2025, May 18). _Philippines' bridges are underdesigned for longevity_ . <u>https://proforale.info/post/philippines-bridges-are-underdesigned-for-longivity</u> 

- Philippine Statistics Authority. (2024). _Total value of damaged infrastructures during major natural events and disasters in the Philippines from 2012 to 2023_ . Statista. <u>https://www.statista.com/statistics/1264083/philippines-value-of-damaged-infrastructures</u> -during-calamities/ 

- Shukla, K., Jagtap, A. D., & Karniadakis, G. E. (2021). Parallel physics-informed neural networks via domain decomposition. _Journal of Computational Physics, 447_ , 110683. <u>https://doi.org/10.1016/j.jcp.2021.110683</u> 

- Statista. (2024). _Natural disasters in the Philippines — statistics & facts_ . <u>https://www.statista.com/topics/5845/natural-disasters-in-the-philippines-at-a-glance/</u> 

- Verified Market Reports. (2024). _Structural simulation software market size and forecast_ . <u>https://www.verifiedmarketreports.com/product/structural-simulation-software-market/</u> 

22 

_Title Proposal: ATLAS_ 

- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. _SIAM Journal on Scientific Computing, 43_ (5), A3055–A3081. https://doi.org/10.1137/20M1318043 

- Xu, C., Cao, B. T., Yuan, Y., & Meschke, G. (2023). Transfer learning based physics-informed neural networks for solving inverse problems in engineering structures under different loading scenarios. _Computer Methods in Applied Mechanics and Engineering, 405_ , 115852. https://doi.org/10.1016/j.cma.2022.115852 

**APPENDICES** 

## **APPENDIX A:** 

Concept 1 



<!-- Start of picture text -->
om atoms ones oes EEEED<br>est 34% 3.44% 18.7% 9.8 min ae,<br>a a<br>ey rae —— ras ay a<br>one<br>a — O mn<br>a Eo<br>oma “ Vom Mises Stress @ vm nae “Shape Optimization — Mote Rass me mon aining Convergence el ? — —<br>maomees Cr) | NN- omO mene<br>ee — Se ees =<br>oe Se QO roveaaton<br><!-- End of picture text -->

Concept 2 

23 

_Title Proposal: ATLAS_ 



<!-- Start of picture text -->
[cssreenter sate wt arvana = Pam hte te inten = Pyne tx en<br><!-- End of picture text -->

## Concept 3 



<!-- Start of picture text -->
Bs Ommmnmsme —Qrenisone rast Queda @ vs = a= ED<br>o>... 0 Om. ° om... ° o>. ° Ot. °<br><!-- End of picture text -->

24 

_Title Proposal: ATLAS_ **APPENDIX B:** ‘e 

25 

_Title Proposal: ATLAS_ 

## **APPENDIX C:** 



<!-- Start of picture text -->
Design the ATLAS<br>Build the XPINN<br>+ framework<br>Collect input data ‘subdomain<br>+ Geometry decomposition strategy<br>+ Boundary‘+ Material+ Loadcases properties conditions ———> continuity+ Set«Define up theinterfaceconditions inverse.<br>design module<br>using automatic<br>differentiation<br>Apply domain |<br>+ UseTrain physics-based the model + Spldecomposition2-6 the subdomains structure into prototypeDevelop+ Input inthe modulePyTorchPython CosaIE<br>‘equations and constraints. + Assign aneural ¢——— + XPINN solver module indings<br>+ Minimize prediction and network to each + Inverse design module<br>interface errors subdomain + FEM validation module<br>+ Enforce continuity at + Visualization module<br>interfaces Create visualizations and<br>. paniapesteaute<br>Fnthe forvardana Evaluate prototype results Reelots.eeement,<br>NO iis the oss YES Cleatpredictions «ConvergenceACEC, MaceoT wolutionee,<br>‘converged? * Displacement+ Stress fieldfield * +. Computational Processing time resource + ATLAS vs FEM comparison<br>‘Von Mises stress usage YES |<br>Improveesand retestmode inverse «NO. << Doresute meet<br>"Compare+ Compare ATLASATLAS outputcup<br>WIRFEM souton YE Eohieten _NO_ eee Sila aril<br>Sepa ee<br>= + Shape optimization + Material usage<br>+ Design quality<br>‘Compare with single- YES<br>Retain with No YES ——_+Checkdemain accuracyFIN No<br>‘adjusted = ¢———— Is Ferror<5%? +<br>parameters + Check+ CheckCheck convergence. processing computational time———> CoSited ———») ie<br>efficiency<br><!-- End of picture text -->

26 

