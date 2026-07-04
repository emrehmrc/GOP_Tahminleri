import numpy as np
import pandas as pd

class SmartFeatureEngineer:
    def __init__(self, df):
        self.df = df

    def add_smart_neighbor_feature(self, target_col, temp_col_candidates=None):
        """
        Geçmişe bakar ve bugüne profil (Ramazan, Haftasonu vb.) olarak uyan,
        sıcaklık olarak da en yakın olan günün tüketimini 'Smart_Lag_Load' olarak ekler.
        """
        print("[SmartFeatureEngineer] 'Profil Bazlı Akıllı Lag' (Smart Neighbor) hesaplanıyor...")
        
        # 1. Referans Alınacak Sıcaklık Sütununu Bul
        selected_temp_col = None
        
        # Kullanıcı aday sütun verdiyse kontrol et
        if temp_col_candidates:
            for col in temp_col_candidates:
                if col in self.df.columns:
                    selected_temp_col = col
                    break
        
        # Bulamazsa 'Hissedilen' geçen herhangi bir sütunu al (Yedek Plan)
        if selected_temp_col is None:
            potential = [c for c in self.df.columns if 'Hissedilen' in c or 'Temperature' in c]
            if potential:
                selected_temp_col = potential[0]

        if selected_temp_col is None:
            print("   -> UYARI: Referans sıcaklık sütunu bulunamadı! Smart Lag atlanıyor.")
            return self.df
        else:
            print(f"   -> Referans Sıcaklık Sütunu: {selected_temp_col}")

        # ---------------------------------------------------------
        # 2. GÜN PROFİLİ OLUŞTURMA (PROFILE ID) 🆔
        # ---------------------------------------------------------
        # Her güne bir "Kimlik Numarası" veriyoruz.
        # Sadece aynı kimliğe sahip günler birbirinin yerine geçebilir.
        
        # Flagleri güvenli bir şekilde al (Yoksa 0 kabul et)
        is_weekend = self.df['Haftanın_Günü'].isin([5, 6]).astype(int) if 'Haftanın_Günü' in self.df.columns else 0
        
        # Senin veri setindeki flag isimlerine göre:
        is_ramadan = self.df['Is_Ramadan'].astype(int) if 'Is_Ramadan' in self.df.columns else 0
        is_holiday = self.df['Milli_Bayram'].astype(int) if 'Milli_Bayram' in self.df.columns else 0
        is_kurban  = self.df['Kurban_Bayram'].astype(int) if 'Kurban_Bayram' in self.df.columns else 0
        
        # Profil ID Oluşturma (Matematiksel kodlama)
        # Örnek: Normal Hafta İçi = 0, Ramazan Hafta Sonu = 11, Kurban Bayramı = 1000...
        self.df['Day_Profile_ID'] = (
            is_weekend * 1 + 
            is_ramadan * 10 + 
            is_holiday * 100 +
            is_kurban  * 1000
        )
        
        # ---------------------------------------------------------
        # 3. GENİŞLETİLMİŞ ARAMA PENCERESİ 🔍
        # ---------------------------------------------------------
        # Ramazan veya Bayram gibi nadir olaylar için 7 gün yetmez.
        # Geçmiş 28 güne (4 Hafta) bakıyoruz.
        lags_to_check = [24*i for i in range(1, 29)] # [24, 48, ... 672]
        
        # Başlangıç Değeri: En güvenli liman "Geçen Haftanın Aynı Saati" (Lag168)
        # Eğer profil eşleşmesi bulamazsak en azından bunu kullansın.
        if 'ADM_Dağıtılan_Enerji_(MWh)_Lag168h' in self.df.columns:
            self.df['Smart_Lag_Load'] = self.df['ADM_Dağıtılan_Enerji_(MWh)_Lag168h']
        else:
            self.df['Smart_Lag_Load'] = self.df[target_col].shift(168)
            
        # Başlangıçta sıcaklık farkını sonsuz yapıyoruz
        self.df['Min_Temp_Diff'] = 9999.0
        
        # Döngü: Geçmiş günleri tara
        for lag in lags_to_check:
            # Geçmiş veriler
            past_temp = self.df[selected_temp_col].shift(lag)
            past_load = self.df[target_col].shift(lag)
            past_profile = self.df['Day_Profile_ID'].shift(lag)
            
            # A. Profil Uyumu (ZORUNLU KURAL)
            # Bugünün profili (örn: Ramazan+Haftasonu) adayın profiliyle AYNI olmalı.
            profile_match = (self.df['Day_Profile_ID'] == past_profile)
            
            # B. Sıcaklık Farkı (TERCİH SEBEBİ)
            current_temp_diff = (self.df[selected_temp_col] - past_temp).abs()
            
            # SEÇİM: Profili uyanlar arasında, sıcaklığı en yakın olanı seç
            better_match = profile_match & (current_temp_diff < self.df['Min_Temp_Diff'])
            
            # Güncelle
            self.df.loc[better_match, 'Min_Temp_Diff'] = current_temp_diff[better_match]
            self.df.loc[better_match, 'Smart_Lag_Load'] = past_load[better_match]

        # Temizlik
        self.df.drop(columns=['Min_Temp_Diff', 'Day_Profile_ID'], inplace=True)
        print("   -> 'Smart_Lag_Load' özelliği başarıyla eklendi.")
        
        return self.df