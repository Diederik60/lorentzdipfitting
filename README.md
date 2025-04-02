Investigating imporvement possibilities for fitting Lorentzian dips in ODMR. From 2D ODMR scan data, frequency sweeps are performed at every pixel. In these frequency sweeps, the photoluminescence is measured as a function of frequency. These sweeps show dips in the frequency, and that’s where we are mostly interested. Typically, the sweep spectra show 2 dips but may display up to 8 dips or clusters of dips. These dips have to be fitted. The dips are Lorentzian dips, and the fitting may fail when there are multiple dips, and you don’t know how many to expect.

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
