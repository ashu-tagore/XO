# XO Project - Feature Documentation

## Physics-Based Engineered Features

### Habitable Zone Features
- `stellar_luminosity`: Stellar luminosity relative to Sun
- `hz_inner/hz_outer`: Habitable zone boundaries (AU)
- `hz_position`: Planet distance relative to HZ center
- `in_habitable_zone`: Boolean flag for HZ membership

### Earth Similarity Index
- `esi_radius/mass/temperature`: Individual ESI components
- `esi_surface`: Combined radius + temperature ESI

### Atmospheric Features
- `escape_velocity_ratio`: Relative to Earth's escape velocity
- `atmospheric_retention`: Qualitative retention capability

### Habitability Score
- `habitability_score`: Weighted composite score (0-1)
