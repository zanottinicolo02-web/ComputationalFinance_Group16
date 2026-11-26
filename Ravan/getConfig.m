function cfg = getConfig(mode)
% GETCONFIG  Single place for all run settings. mode = 'train' or 'test'.
    switch mode
        case 'train'    % collar issued 06-Jan-2020, averaging Feb 2021, legs Apr21 / May21
            cfg.parquetFile = 'unpivoted_bid_offer_train_clean.parquet';
            cfg.excelFile   = 'COJ21_COZ22_historical_series.xlsx';
            cfg.leg1Code = 'COJ21';  cfg.leg2Code = 'COK21';
            cfg.avgYear  = 2021;     cfg.avgMonth = 2;
            cfg.leg1Year = 2021;     cfg.leg1Month = 4;       % Apr21 contract month
        case 'test'     % collar issued 06-Apr-2021, averaging Oct 2022, legs Dec22 / Jan23
            cfg.parquetFile = '<TEST_PARQUET_FILE>';           % fill in with the released test files
            cfg.excelFile   = '<TEST_EXCEL_FILE>';
            cfg.leg1Code = 'COZ22';  cfg.leg2Code = 'COF23';
            cfg.avgYear  = 2022;     cfg.avgMonth = 10;
            cfg.leg1Year = 2022;     cfg.leg1Month = 12;      % Dec22 contract month
        otherwise
            error('getConfig:mode', 'mode must be ''train'' or ''test''.');
    end
    cfg.n2RollDays = 1;     % Annex B + Q&A: last fixing rolls to leg 2 (Apr21/May21 and Dec22/Jan23)

    % Strategy parameters, identical for both runs
    cfg.hedge = struct('dynamics', 'sticky_strike', ...    % or 'sticky_moneyness'
                       'useGamma', false, ...              % vega-only hedge with one listed option
                       'qTol', 0.6, 'band1', 0.35, 'keepStrikeAcrossType', true);
    cfg.optsIV  = struct();    % overrides for extractImpliedVols filters
    cfg.optsRho = struct();    % overrides for estimateCorrelation
    cfg.compare = false;       % true: runPipeline also runs futures-only and unhedged
end
