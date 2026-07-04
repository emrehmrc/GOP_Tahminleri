import pandas as pd
import numpy as np
import os 
import sys
from src.smart_features import SmartFeatureEngineer

from config_live import (
    INPUT_FILE_PATH, 
    RAW_TARGET_COL, 
    RAW_DATE_COL, 
    RAW_HOUR_COL, 
    COLS_TO_DROP,
    WARMUP_PERIOD, 
    TEST_SIZE,
    DATA_START_DATE,
    DATA_END_DATE
)

class DataManager:
    def __init__(self):
        self.data = None
        self.RAW_TARGET_COL = RAW_TARGET_COL
        
    def load_and_preprocess(self):
        """
        Veriyi yükler, tarih ayarını yapar, kategorik dönüşümü yapar.
        """
        parquet_path = INPUT_FILE_PATH.replace('.xlsx', '.parquet')
        
        # Eğer parquet dosyası var ve xlsx'ten daha güncelse, oradan hızlıca yükle
        if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) > os.path.getmtime(INPUT_FILE_PATH):
            print(f"[DataManager] Hızlı yükleme: {parquet_path} okunuyor...")
            df = pd.read_parquet(parquet_path)
        else:
            print(f"[DataManager] Veri Excel'den yükleniyor (bir kaç dakika sürebilir): {INPUT_FILE_PATH}")
            df = pd.read_excel(INPUT_FILE_PATH, engine='openpyxl')
            
            # Sonraki sefer çok daha hızlı yüklenebilmesi için Parquet olarak kaydet
            print(f"[DataManager] Veri Parquet formatında önbelleğe alınıyor (Hızlı yükleme için)...")
            df.to_parquet(parquet_path, engine='pyarrow')


        # 2. Sayısal Dönüşümler (Virgül -> Nokta)
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Tarih ve Özel Gün dışındakileri sayıya çevir
            if col in [RAW_DATE_COL, 'ÖzelGün_Adı']:
                continue
            try:
                # Varsa virgülleri noktaya çevirip float yap
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except (ValueError, AttributeError):
                pass



        # ----------------------------------------------------
        # 3. Tarih ve Saat İşlemleri
        df[RAW_DATE_COL] = pd.to_datetime(df[RAW_DATE_COL])

        # --- [YENİ] ZAMANSAL FİLTRELEME ---
        # Config'den gelen string tarihleri datetime objesine çevirerek karşılaştıralım
        if DATA_START_DATE:
            start_dt = pd.to_datetime(DATA_START_DATE)
            df = df[df[RAW_DATE_COL] >= start_dt]
            print(f"[DataManager] Filtre Uygulandı: {DATA_START_DATE} sonrası veriler alınıyor.")

        if DATA_END_DATE:
            end_dt = pd.to_datetime(DATA_END_DATE)
            df = df[df[RAW_DATE_COL] < end_dt] # Belirtilen tarihe kadar (o gün dahil değil)
            print(f"[DataManager] Filtre Uygulandı: {DATA_END_DATE} öncesi veriler alınıyor.")
        # ---------------------------------
        
        # Tam datetime index oluşturma
        # Excel'deki Saat=0 verisi, günün 24. saatini (23:00-00:00) temsil eder.
        # Bu kronolojik olarak ertesi günün 00:00'ına denk gelir.
        corrected_hours = df[RAW_HOUR_COL].replace(0, 24)
        df['Datetime'] = df[RAW_DATE_COL].dt.normalize() + pd.to_timedelta(corrected_hours, unit='h')
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True) # Tarih sırasını garantiye al

        # Zaman bileşenlerinin INT olduğundan emin ol (XGBoost için kritik)
        time_features = ['Yıl', 'Ay', 'Gün', 'Saat', 'Haftanın_Günü', "Ramazan_Bayram","Kurban_Bayram"]
        for tf in time_features:
            if tf in df.columns:
                df[tf] = df[tf].astype(int)
        
        
        # ----------------------------------------------------
        # [ADIM 2] OKUL TATİLLERİNİ AYRIŞTIRMA (Yaz vs Kış)
        # ----------------------------------------------------
        print("[DataManager] Sömestr ve Yaz tatilleri işleniyor...")

        # 1. SÖMESTR TATİLLERİ (Kış Karakteristiği)
        semester_ranges = [
            ('2018-01-22', '2018-02-04'), ('2019-01-21', '2019-02-03'),
            ('2020-01-20', '2020-02-02'), ('2021-01-25', '2021-02-14'),
            ('2022-01-24', '2022-02-06'), ('2023-01-23', '2023-02-19'), # Deprem dahil
            ('2024-01-22', '2024-02-04'), ('2025-01-20', '2025-02-02'),
            ('2026-01-26', '2026-02-08'),
        ]
        
        df['Is_Semester'] = 0
        for start, end in semester_ranges:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'Is_Semester'] = 1

        # 2. YAZ TATİLLERİ (Turizm/Klima Karakteristiği)
        summer_ranges = [
            ('2018-06-09', '2018-09-16'), ('2019-06-15', '2019-09-08'),
            ('2020-03-16', '2020-09-20'), 
            ('2021-07-03', '2021-09-05'), ('2022-06-18', '2022-09-11'),
            ('2023-06-17', '2023-09-10'), ('2024-06-15', '2024-09-08'),
            ('2025-06-21', '2025-09-08')
        ]

        df['Is_Summer_Break'] = 0
        for start, end in summer_ranges:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'Is_Summer_Break'] = 1
            
        df['Is_Semester'] = df['Is_Semester'].astype(int)
        df['Is_Summer_Break'] = df['Is_Summer_Break'].astype(int)



        
        # ----------------------------------------------------
        # 4. YENİ ÖZELLİK MÜHENDİSLİĞİ (Gelişmiş Hava Durumu & Aggregation)
        # ----------------------------------------------------
        print("[DataManager] İl bazlı sıcaklık ortalamaları ve termal özellikler hesaplanıyor...")

        # A. İL BAZLI ORTALAMALAR (AGGREGATION)
        # Excel'deki yeni kısaltmalarınıza göre map'i güncelledik
        province_map = {
            'MUGLA': 'Hissedilen_Sıcaklık_Mean_MUGLA',
            'DENIZLI': 'Hissedilen_Sıcaklık_Mean_DNZ', # 'DNZ' yerine 'DENIZLI' yaptık
            'AYDIN':   'Hissedilen_Sıcaklık_Mean_AYD'  # 'AYD' yerine 'AYDIN' yaptık
        }

        cols_to_remove = []

        for province_code, new_col_name in province_map.items():
            # ARAMA KRİTERİ: 'Hissedilen_Sıcaklık' yerine 'app_temp_actual' arıyoruz
            relevant_cols = [c for c in df.columns if province_code in c and 'app_temp_actual' in c]
            
            if relevant_cols:
                df[new_col_name] = df[relevant_cols].mean(axis=1)
                # Orijinal sütunları silebiliriz (opsiyonel)
                # cols_to_remove.extend(relevant_cols) 
                print(f"   -> {new_col_name} oluşturuldu ({len(relevant_cols)} istasyon birleştirildi).")

        # ... (BASE_TEMP_COL seçimi aynı kalabilir) ...
        available_means = [
            'Hissedilen_Sıcaklık_Mean_MUGLA', 
            'Hissedilen_Sıcaklık_Mean_DNZ', 
            'Hissedilen_Sıcaklık_Mean_AYD'
        ]
        
        BASE_TEMP_COL = next((col for col in available_means if col in df.columns), None)
        
        # ÇÖKMEYİ ENGELLEYEN KRİTİK DEĞİŞİKLİK:
        # HDD ve CDD hesaplamalarını da bu 'if' bloğunun içine almalıyız
        if BASE_TEMP_COL:
            print(f"   -> Termal özellikler için baz alınan sütun: {BASE_TEMP_COL}")
            
            df['Temp_Squared_18'] = (df[BASE_TEMP_COL] - 18) ** 2
            
            for lag in [3, 6, 12]:
                df[f'Temp_Lag{lag}h'] = df[BASE_TEMP_COL].shift(lag)
            
            df['Temp_Diff_24h'] = df[BASE_TEMP_COL] - df[BASE_TEMP_COL].shift(24)
            df['Temp_Diff_3h'] = df[BASE_TEMP_COL].diff(3)

            # Isıtma/Soğutma streslerini buraya taşıdık ki BASE_TEMP_COL None ise çökmesin
            df['HDD_Heating_Stress'] = np.maximum(0, 16 - df[BASE_TEMP_COL])
            df['CDD_Cooling_Stress'] = np.maximum(0, df[BASE_TEMP_COL] - 24)
            df['Extreme_Heat_Impact'] = df['CDD_Cooling_Stress'] ** 2
            df['Extreme_Cold_Impact'] = df['HDD_Heating_Stress'] ** 2

            print("   -> Temp_Squared, HDD/CDD ve Delta özellikleri başarıyla eklendi.")
        else:
            print("   -> HATA: Sıcaklık sütunları bulunamadı! Lütfen Excel başlıklarını kontrol edin.")
    
        # --- GELİŞMİŞ HAVA VE GÜNEŞ ÖZELLİKLERİ ---
        from config_live import ENABLE_THERMAL_FEATURES
        if ENABLE_THERMAL_FEATURES:
            from src.thermal_features import add_thermal_features
            df = add_thermal_features(df)
            
        # Solar generation shaving proxy (GHI * clear sky probability)
        if 'GHI_ADM_Weighted' in df.columns:
            cloud_cols = [c for c in df.columns if c.endswith('_cloud_actual')]
            cloud_mean = df[cloud_cols].mean(axis=1) if cloud_cols else 0.0
            df['Solar_Shaving_Proxy'] = df['GHI_ADM_Weighted'] * (1.0 - (cloud_mean / 100.0))
            print("   -> Solar_Shaving_Proxy created.")
        
        # 5. Kategorik Veri İşleme
        # Özel Günler -> Category
        if 'ÖzelGün_Adı' in df.columns:
            print("[DataManager] Converting 'ÖzelGün_Adı' to category...")
            df['ÖzelGün_Adı'] = df['ÖzelGün_Adı'].astype('category')


        binary_flags = ['Is_Ramadan', 'Is_Sahur', 'Is_lockdown','Yilbasi', "weekday_after_bayram", "is_religional_holiday", "before_yilbasi", "weekday_after_yilbasi","Secim_Gunu", "Milli_Bayram"]

        print(f"[DataManager] Binary sütunlar (0/1) işleniyor: {binary_flags}")

        for col in binary_flags:
            if col in df.columns:
                # 1. Her ihtimale karşı eksik varsa 0 yap (Sen eksik yok dedin ama güvenliktir)
                df[col] = df[col].fillna(0)
                
                # 2. Tipini 'int' (Tamsayı) yap.
                df[col] = df[col].astype(int)
            else:
                print(f"   -> UYARI: '{col}' sütunu Excel'de bulunamadı! İsmi doğru yazdın mı?")

        """
        # ----------------------------------------------------------------
        # OUTLIER TEMİZLİĞİ (LAGLERDEN ÖNCE YAPILMALI!)
        # ----------------------------------------------------------------
        self.clean_seasonal_outliers(window_days=14, threshold=3.5)

        """

        from config_live import ENABLE_HOLIDAY_LAG_CLEAN
        target_clean = df[RAW_TARGET_COL].copy()

        if ENABLE_HOLIDAY_LAG_CLEAN:
            print("[DataManager] Holiday Lag Cleaning is active. Cleaning target lag features...")
            from src.holiday_calendar import build_holiday_calendar, is_holiday_hour_for_lag
            
            # Build holiday calendar
            years = list(range(df.index.year.min() - 1, df.index.year.max() + 2))
            cal = build_holiday_calendar(years)
            
            # Gelişmiş lag temizliği ve bayram sonrası özelliklerini ekleme
            from src.holiday_lag_clean import attach_lag_clean_and_post_holiday
            df = attach_lag_clean_and_post_holiday(df, RAW_TARGET_COL, cal)
            
            # Find holiday timestamps
            holiday_mask = [is_holiday_hour_for_lag(ts, cal) for ts in df.index]
            holiday_ts = df.index[holiday_mask]
            
            print(f"   -> Found {len(holiday_ts)} holiday hours out of {len(df)} total hours.")
            
            # Replace holiday values with the closest preceding normal weekday hour
            for ts in holiday_ts:
                ref_ts = ts - pd.Timedelta(hours=24)
                while ref_ts in target_clean.index and (is_holiday_hour_for_lag(ref_ts, cal) or ref_ts.dayofweek >= 5):
                    ref_ts -= pd.Timedelta(hours=24)
                
                if ref_ts in target_clean.index:
                    target_clean.loc[ts] = target_clean.loc[ref_ts]
            
            # Overwrite lag features using target_clean
            df[f'{RAW_TARGET_COL}_Lag24h'] = target_clean.shift(24)
            df[f'{RAW_TARGET_COL}_Lag25h'] = target_clean.shift(25)
            df[f'{RAW_TARGET_COL}_Lag26h'] = target_clean.shift(26)
            df[f'{RAW_TARGET_COL}_Lag27h'] = target_clean.shift(27)
            df[f'{RAW_TARGET_COL}_Lag168h'] = target_clean.shift(168)
            df[f'{RAW_TARGET_COL}_Lag336h'] = target_clean.shift(336)
            df[f'{RAW_TARGET_COL}_Lag504h'] = target_clean.shift(504)
            
            # Overwrite rolling average of lag
            df['Rolling_Mean_3h_Lag24h'] = target_clean.shift(24).rolling(window=3).mean()
            print("   -> Lag features cleaned successfully.")
        else:
            print("[DataManager] Holiday Lag Cleaning is disabled.")

        # Bayram Recovery Transition Lag Substitution
        if "post_holiday_recovery_lag_24" in df.columns:
            post_hol_mask = (df.get("is_post_holiday_day1", 0) == 1) | (df.get("weekday_after_bayram", 0) == 1)
            if post_hol_mask.any():
                print(f"[DataManager] Replacing polluted Lag24h with post_holiday_recovery_lag_24 for {post_hol_mask.sum()} hours.")
                df.loc[post_hol_mask, f'{RAW_TARGET_COL}_Lag24h'] = df.loc[post_hol_mask, 'post_holiday_recovery_lag_24']
                if 'lag_24_clean' in df.columns:
                    df.loc[post_hol_mask, 'lag_24_clean'] = df.loc[post_hol_mask, 'post_holiday_recovery_lag_24']
                
        if "post_holiday_recovery_lag_168" in df.columns:
            post_hol_mask = (df.get("is_post_holiday_day1", 0) == 1) | (df.get("weekday_after_bayram", 0) == 1)
            if post_hol_mask.any():
                print(f"[DataManager] Replacing polluted Lag168h with post_holiday_recovery_lag_168 for {post_hol_mask.sum()} hours.")
                df.loc[post_hol_mask, f'{RAW_TARGET_COL}_Lag168h'] = df.loc[post_hol_mask, 'post_holiday_recovery_lag_168']
                if 'lag_168_clean' in df.columns:
                    df.loc[post_hol_mask, 'lag_168_clean'] = df.loc[post_hol_mask, 'post_holiday_recovery_lag_168']

        # Re-compute Mean_Last_3_Days_Same_Hour using target_clean
        df['Mean_Last_3_Days_Same_Hour'] = (
            target_clean.shift(24) + 
            target_clean.shift(48) + 
            target_clean.shift(72)
        ) / 3

        # Monday transition / Last Workday Lag feature
        print("[DataManager] Creating Last_Workday_Lag feature...")
        dow = target_clean.index.dayofweek
        last_workday_lag = target_clean.shift(24).copy()
        
        # Overwrite Monday (0) with shift 72h (Friday)
        monday_mask = dow == 0
        last_workday_lag.loc[monday_mask] = target_clean.shift(72).loc[monday_mask]
        
        # Overwrite Sunday (6) with shift 48h (Friday)
        sunday_mask = dow == 6
        last_workday_lag.loc[sunday_mask] = target_clean.shift(48).loc[sunday_mask]
        
        df['Last_Workday_Lag'] = last_workday_lag
        print("   -> Last_Workday_Lag created.")

        # ----------------------------------------------------
        # 6a. A-FAMILY FEATURES (A1: momentum, A2: volatility, A3: ratio)
        # ----------------------------------------------------
        from config_live import ENABLE_A1_FEATURES, ENABLE_A2_FEATURES, ENABLE_A3_FEATURES
        lag24 = f'{RAW_TARGET_COL}_Lag24h'
        lag48 = f'{RAW_TARGET_COL}_Lag24h'  # will be shift(48) not shift(24)
        lag72 = f'{RAW_TARGET_COL}_Lag24h'
        lag168 = f'{RAW_TARGET_COL}_Lag168h'

        if ENABLE_A1_FEATURES:
            print("[DataManager] Adding A1 momentum features...")
            # Load_Chg_24h = Lag24h - Lag48h
            df['Load_Chg_24h'] = target_clean.shift(24) - target_clean.shift(48)
            # Load_Chg_168h = Lag24h - Lag168h
            df['Load_Chg_168h'] = target_clean.shift(24) - target_clean.shift(168)
            # Load_Dev_3d = Lag24h - Mean_Last_3_Days
            if 'Mean_Last_3_Days_Same_Hour' in df.columns:
                df['Load_Dev_3d'] = target_clean.shift(24) - df['Mean_Last_3_Days_Same_Hour']
            print("   -> A1 features added.")

        if ENABLE_A2_FEATURES:
            print("[DataManager] Adding A2 volatility features...")
            # Load_Vol_3d = std(Lag48h, Lag72h, Lag96h)
            l48 = target_clean.shift(48)
            l72 = target_clean.shift(72)
            l96 = target_clean.shift(96)
            df['Load_Vol_3d'] = pd.concat([l48, l72, l96], axis=1).std(axis=1)
            # Load_Range_3d = max - min(Lag48h, Lag72h, Lag96h)
            df['Load_Range_3d'] = pd.concat([l48, l72, l96], axis=1).max(axis=1) - pd.concat([l48, l72, l96], axis=1).min(axis=1)
            # Load_Vol_1w = std(Lag48h..Lag192h, 7 points)
            lags_1w = [target_clean.shift(h) for h in [48, 72, 96, 120, 144, 168, 192]]
            df['Load_Vol_1w'] = pd.concat(lags_1w, axis=1).std(axis=1)
            print("   -> A2 features added.")

        if ENABLE_A3_FEATURES:
            print("[DataManager] Adding A3 ratio features...")
            eps = 1.0
            # Load_Ratio_3d = Lag24h / (Mean_Last_3_Days + 1)
            if 'Mean_Last_3_Days_Same_Hour' in df.columns:
                df['Load_Ratio_3d'] = target_clean.shift(24) / (df['Mean_Last_3_Days_Same_Hour'] + eps)
            # Load_Ratio_168h = Lag24h / (Lag168h + 1)
            df['Load_Ratio_168h'] = target_clean.shift(24) / (target_clean.shift(168) + eps)
            # Load_Ratio_Workday = Lag24h / (Last_Workday_Lag + 1)
            if 'Last_Workday_Lag' in df.columns:
                df['Load_Ratio_Workday'] = target_clean.shift(24) / (df['Last_Workday_Lag'] + eps)
            print("   -> A3 features added.")


        # ----------------------------------------------------
        # 6b. AKILLI KOMŞU ÖZELLİĞİ (SMART NEIGHBOR FEATURE)
        # ----------------------------------------------------
        """
        smart_engineer = SmartFeatureEngineer(df)
        
        # Aday sıcaklık sütunlarını verelim (Varsa ortalamayı, yoksa Menteşe'yi kullanır)
        temp_candidates = [
            'Hissedilen_Sıcaklık_Mean_MUGLA', 
            'Hissedilen_Sıcaklık-MUGLA_MenteseCenter_OpenMeteo'
        ]
        
        # İşlemi yap ve df'i güncelle
        df = smart_engineer.add_smart_neighbor_feature(
            target_col=RAW_TARGET_COL, 
            temp_col_candidates=temp_candidates
        )
        """


        
        # 6. Config'den Gelen Gereksiz Sütunları Atma
        if COLS_TO_DROP:
            print(f"[DataManager] Dropping columns from config: {COLS_TO_DROP}")
            existing_drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
            df.drop(columns=existing_drop_cols, inplace=True)

        

        # 7. Eksik Verileri Çıkarma
        print(f"[DataManager] Dropping warm-up period ({WARMUP_PERIOD} rows)...")
        df = df.iloc[WARMUP_PERIOD:] 
        df.dropna(inplace=True)

        self.data = df
        print(f"[DataManager] Preprocessing complete. Shape: {df.shape}")
        
        # Veri tiplerini kontrol için yazdır
        print(df.dtypes)
        
        return self.data
    
    """
    def clean_seasonal_outliers(self, window_days=14, threshold=3.5):
        
       
        #  Zeki Outlier Temizliği:
        # 1. Sadece 'Normal' günlerdeki teknik hataları temizler.
        #2. Özel günlere (Bayram, Ramazan vb.) DOKUNMAZ.
        
        print(f"[DataManager] Mevsimsel Outlier temizliği yapılıyor (Özel Günler Korumalı)...")
        
        col = self.RAW_TARGET_COL
        
        # --- ADIM 1: DOKUNULMAZLIK LİSTESİ OLUŞTUR ---
        # Bu günlerde tüketim ne kadar saparsa sapsın, bu bir veri hatası değil,
        # modelin öğrenmesi gereken bir 'davranış'tır.
        
        special_day_cols = [
            'Is_Ramadan', 'Ramazan_Bayram', 'Kurban_Bayram', 
            'Milli_Bayram', 'Is_Sahur', 'Yilbasi', 'Secim_Gunu', 'Is_lockdown'
        ]
        
        # Başlangıçta kimse korumalı değil (Hepsi False)
        is_protected = pd.Series(False, index=self.data.index)
        
        for p_col in special_day_cols:
            if p_col in self.data.columns:
                # Eğer o sütunda 1 varsa, o satır korumalıdır
                is_protected |= (self.data[p_col] == 1)

        # Haftasonlarına da dokunmasın (Opsiyonel ama önerilir)
        if 'Haftanın_Günü' in self.data.columns:
             # Eğer kategori ise koduna bakmak lazım ama genelde sayısal çevirmiştik
             # Veya hafta sonu flag'in varsa onu kullan. Yoksa şimdilik kalsın.
             pass

        print(f"   -> {is_protected.sum()} saatlik veri 'Özel Gün' olduğu için korumaya alındı.")

        # --- ADIM 2: ROLLING Z-SCORE HESABI ---
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_days)
        grouped = self.data.groupby('Saat')[col]
        
        local_mean = grouped.transform(lambda x: x.shift(1).rolling(window=window_days, min_periods=7).mean())
        local_std = grouped.transform(lambda x: x.shift(1).rolling(window=window_days, min_periods=7).std())
        
        z_score = (self.data[col] - local_mean) / local_std
        
        # --- ADIM 3: AKILLI FİLTRELEME ---
        # Bir verinin outlier sayılması için 2 şart lazım:
        # 1. Z-Score eşiği geçmiş olmalı (Anormal olmalı)
        # 2. VE Korumalı bir gün OLMAMALI (Normal bir gün olmalı)
        
        raw_outlier_mask = np.abs(z_score) > threshold
        
        # İşte sihirli satır burası:
        final_outlier_mask = raw_outlier_mask & (~is_protected)
        
        outlier_count = final_outlier_mask.sum()
        ignored_count = raw_outlier_mask.sum() - outlier_count
        
        if outlier_count > 0:
            print(f"   -> {outlier_count} adet teknik outlier tespit edildi ve temizlendi.")
            print(f"   -> {ignored_count} adet anormallik 'Özel Gün' olduğu için SİLİNMEDİ (Doğrusu bu).")
            
            # Tamir Et
            self.data.loc[final_outlier_mask, col] = np.nan
            self.data[col] = self.data[col].interpolate(method='time')
        else:
            print("   -> Temiz. Müdahale edilecek outlier bulunamadı.")
            
        return self.data
         """

    def get_train_test_split(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        # Test seti son n satır
        split_idx = len(self.data) - TEST_SIZE
        train_df = self.data.iloc[:split_idx]
        test_df = self.data.iloc[split_idx:]

        print(f"[DataManager] Split Done. Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Feature Selection
        # number: int, float | category: category | bool: bool
        feature_cols = self.data.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
        
        if RAW_TARGET_COL in feature_cols:
            feature_cols.remove(RAW_TARGET_COL)
            
        print(f"[Features Used]: {feature_cols}")

        X_train = train_df[feature_cols]
        y_train = train_df[RAW_TARGET_COL]
        X_test = test_df[feature_cols]
        y_test = test_df[RAW_TARGET_COL]

        return X_train, y_train, X_test, y_test