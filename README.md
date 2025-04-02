This project investigates improvements for fitting Lorentzian dips in Optically Detected
Magnetic Resonance (ODMR) spectra from NV centers in diamond. Two algorithms were
developed: the Bimodal Lorentzian Spectral Fitting Algorithm (BLSFA) for double-dip
scenarios and the Multimodal Lorentzian Spectral Fitting Algorithm (MLSFA) for complex
multi-dip spectra. Both algorithms implement strategic parameter log-scaling for amplitude
and width variables while maintaining linear frequency parameters, robust initial parameter
estimation based on spectral characteristics, and constrained trust region optimization
methods with the option for Levenberg-Marquardt to enhance convergence reliability. The
BLSFA achieves computational efficiency (≈ 1,500 pixels/second) with success rates
exceeding 99%, effectively eliminating outliers present in previous methods. The MLSFA
extends capabilities to handle multiple dips by implementing specialized weighting systems
that prioritize physically relevant spectral regions during optimization, though at higher
computational cost. A comprehensive graphical interface was developed featuring a pixel
averaging technique that substitutes low-quality pixels with values from surrounding
high-quality pixels, significantly improving visualization while preserving physical significance.
Machine learning approaches (branched neural network and XGBoost) were also explored but
revealed fundamental limitations in capturing the physical interdependencies of Lorentzian
parameters despite extensive hyperparameter optimization. Traditional curve fitting
approaches proved superior, though future hybrid methods may be explored. The algorithms
and visualization tools developed enable more reliable parameter extraction for quantum
sensing applications

## Bimodal Lorentzian Spectral Fitting Algorithm Flowchart

```mermaid

flowchart TB
    Start([Start]) --> PreProc[Preprocess Data\nNormalize & Smooth]
    
    PreProc --> DipDetection
    
    subgraph DipDetection["1. Dip Detection"]
        direction LR
        FindDips[Find Dips via\nMultiple Prominence Levels] --> DigestedDips{Found\ndips?}
        DigestedDips -- 2+ dips --> SortProminence[Sort by Prominence\nSelect Top Two]
        DigestedDips -- 1 dip --> AnalyzeSingle[Analyze Single Dip\nCheck if Merged]
        DigestedDips -- No dips --> DefaultParams[Use Default\nDip Parameters]
        AnalyzeSingle --> IsMerged{Wide\ndip?}
        IsMerged -- Yes --> TreatAsMerged[Treat as\nMerged Dips]
        IsMerged -- No --> TreatAsSingle[Treat as\nSingle Dip]
    end
    
    DipDetection --> ParamSetup
    
    subgraph ParamSetup["2. Parameter Setup"]
        direction LR
        EstimateParams[Estimate Initial\nParameters] --> ConvertLogScale[Convert I₀, A, width\nto Log Scale]
        ConvertLogScale --> SetupBounds[Setup Parameter\nBounds]
    end
    
    ParamSetup --> Optimization
    
    subgraph Optimization["3. Optimization"]
        direction LR
        RunOptimizer[Run Curve Fit\nwith Hybrid Parameters] --> FitSuccess{Good\nfit?}
        FitSuccess -- Yes --> ExtractParams[Extract\nParameters]
        FitSuccess -- No --> RetryFit[Try Again with\nSingle Dip Assumption]
        RetryFit --> SecondCheck{Improved\nfit?}
        SecondCheck -- Yes --> ExtractParams
        SecondCheck -- No --> UseDefaults[Use Default\nValues]
        UseDefaults --> Return
        ExtractParams --> QualityCheck{Quality\n≥ 0.5?}
        QualityCheck -- Yes --> Return
        QualityCheck -- No --> UseDefaults
    end

    Return([Return Results])

    %% Styling
    classDef detailsClass fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef defaultClass fill:#fff,stroke:#999,stroke-width:1px,color:#000
    
    class DipDetection,ParamSetup,Optimization detailsClass
    class DigestedDips,IsMerged,FitSuccess,SecondCheck,QualityCheck defaultClass

```


## Multimodal Lorentzian Spectral Fitting Flowchart

```mermaid

flowchart TB
    Start([Start]) --> PreProc[Preprocess Data\nNormalize & Smooth]
    
    PreProc --> DipDetection
    
    subgraph DipDetection["1. Dip Detection"]
        direction LR
        FindOuterDips[Find Outermost Dips] --> CenterFound{Center\nfound?}
        CenterFound -- Yes --> FindDipPairs[Find All Dips]
        CenterFound -- No --> UseMeanFreq[Use mean\nfrequency]
        UseMeanFreq --> FindDipPairs
    end
    
    DipDetection --> PairFormation
    
    subgraph PairFormation["2. Pair Formation"]
        direction LR
        EnoughDips{Enough\ndips?} -- Yes --> SortDips[Sort into L/R]
        EnoughDips -- No --> CreateSyntheticDips[Create synthetic\ndips]
        CreateSyntheticDips --> SortDips
        
        SortDips --> BalancedPairs{Balanced\npairs?}
        BalancedPairs -- Yes --> FormPairs[Form Dip Pairs]
        BalancedPairs -- No --> GenerateMissingDips[Generate\nmissing dips]
        GenerateMissingDips --> FormPairs
    end
    
    PairFormation --> ParamSetup
    
    subgraph ParamSetup["3. Parameter Setup"]
        direction LR
        EstimateParams[Estimate Initial\nParameters] --> ConvertLogScale[Convert to\nLog Scale]
        ConvertLogScale --> SetupWeights[Setup\nWeighting]
        SetupWeights --> SetupBounds[Setup\nBounds]
    end
    
    ParamSetup --> Optimization
    
    subgraph Optimization["4. Optimization"]
        direction LR
        RunOptimizer[Run Non-Linear\nOptimization] --> FitSuccess{Fit\nsuccess?}
        FitSuccess -- Yes --> ExtractParams[Extract\nParameters]
        FitSuccess -- No --> UseDefaults[Use Default\nValues]
        UseDefaults --> Return
        ExtractParams --> Return([Return\nResults])
    end

    %% Styling
    classDef detailsClass fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef defaultClass fill:#fff,stroke:#999,stroke-width:1px,color:#000
    
    class DipDetection,PairFormation,ParamSetup,Optimization detailsClass
    class EnoughDips,CenterFound,BalancedPairs,FitSuccess defaultClass

```
