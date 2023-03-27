use std;
use num_dual::*;
use plotters::prelude::*;
use root_search::root_search;

struct KronigPenney {
    u1: f32,
    u2: f32,
    d: f32,
    v: f32,
    // w: f32,
    x1: f32,
    x2: f32,
    lower: f32,
    upper: f32,
    resolution: i32,
    patience: i32,
    tolerance: f32,
}

trait RootsFinder {
    fn find_roots(&self) -> Vec<Vec<(f32, f32)>>;
}

fn find_roots_in_parallel<'a, F>(lower: f32, upper: f32, resolution: i32, patience: i32, tolerance: f32, transcendental_equation: F) -> Vec<Vec<(f32, f32)>> 
where F: Fn(DualVec<f32, f32, 1>, f32) -> DualVec<f32, f32, 1> + Copy + Send + Sync + 'a {
    let mut roots: Vec<Vec<(f32, f32)>> = Vec::new();
    for i in 0..resolution {
        let step = 2.0 * i as f32 / (resolution as f32);
        let k = std::f32::consts::PI * (-1.0 + step);
        let root_equation = move |q: DualVec<f32, f32, 1>| transcendental_equation(q, k); 
        let results = root_search(root_equation, lower, upper, resolution, patience, tolerance);

        let mut roots_per_k: Vec<(f32, f32)> = Vec::new();
        for root in &results.0 {
            roots_per_k.push((k, *root));
        }
        roots.push(roots_per_k);
    }
    roots
}


impl RootsFinder for KronigPenney {
    fn find_roots(&self) -> Vec<Vec<(f32, f32)>> {
        if self.lower < 0.0 && self.upper <= 0.0 {
            if self.u1 > 0.0 && self.u2 > 0.0 {
                panic!("No roots for negative energy and positive potential")
            }
            let transcendental_equation = |q: DualVec<f32, f32, 1>, k: f32| self.transcendental_equation_for_negative_energy_bulk_roots(q, k);
            return find_roots_in_parallel(self.lower, self.upper, self.resolution, self.patience, self.tolerance, transcendental_equation);
        }
        else if self.lower >= 0.0 && self.upper > 0.0 {
            if self.u1 < 0.0 && self.u2 < 0.0 {
                panic!("No roots for positive energy and negative potential")
            }
            let transcendental_equation = |q: DualVec<f32, f32, 1>, k: f32| self.transcendental_equation_for_positive_energy_bulk_roots(q, k);
            return find_roots_in_parallel(self.lower, self.upper, self.resolution, self.patience, self.tolerance, transcendental_equation);
        }
        else {
            panic!("Cannot find mixed roots, e.g. bound and propagating");
        }
    }
}


impl KronigPenney {
    fn new(
        u1: f32,
        u2: f32,
        d: f32,
        v: f32,
        lower: f32,
        upper: f32,
        resolution: i32,
        patience: i32,
        tolerance: f32,
    ) -> Self {
        if lower > upper {
            panic!("Lower bound must be less than upper bound");
        }
        // let w = d - v;
        let x1 = -v / 2.0;
        let x2 = v / 2.0;
        Self {
            u1,
            u2,
            d,
            v,
            // w,
            x1,
            x2,
            lower,
            upper,
            resolution,
            patience,
            tolerance,
        }
    }

    fn transcendental_equation_for_positive_energy_bulk_roots(&self, q: DualVec<f32, f32, 1>, k: f32) -> DualVec<f32, f32, 1> {
        let first_term = || -> DualVec<f32, f32, 1> {
            let mut arg = k;
            arg.scale(self.d);
            let mut coeff = q.powi(2);
            coeff.scale(arg.cos());
            coeff
        };

        let second_term = || -> DualVec<f32, f32, 1> {
            let mut diff = self.x2 - self.x1;
            diff.scale(2.0);
            let mut arg = q;
            arg.scale(self.d - (2.0 * self.v));
            let coeff = self.u1 * self.u2;
            arg = arg.cos();
            arg.scale(coeff as f32);
            arg
        };

        let third_term = || -> DualVec<f32, f32, 1> {
            let coeff = q.powi(2) - Dual32::from_re((self.u1 * self.u2) as f32);
            let mut arg = q;
            arg.scale(self.d);
            coeff * arg.cos()
        };

        let fourth_term = || -> DualVec<f32, f32, 1> {
            let mut coeff = q;
            coeff.scale((self.u1 + self.u2) as f32);
            let mut arg = q;
            arg.scale(self.d);
            coeff * arg.sin()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }

    fn transcendental_equation_for_negative_energy_bulk_roots(&self, q: DualVec<f32, f32, 1>, k: f32) -> DualVec<f32, f32, 1> {
        let first_term = || -> DualVec<f32, f32, 1> {
            let mut arg = k;
            arg.scale(self.d);
            let mut coeff = q.powi(2);
            coeff.scale(arg.cos());
            coeff
        };

        let second_term = || -> DualVec<f32, f32, 1> {
            let mut diff = self.x2 - self.x1;
            diff.scale(2.0);
            let mut arg = q;
            arg.scale(self.d - (2.0 * self.v));
            let coeff = - self.u1 * self.u2;
            arg = arg.cosh();
            arg.scale(coeff as f32);
            arg
        };

        let third_term = || -> DualVec<f32, f32, 1> {
            let coeff = q.powi(2) + Dual32::from_re((self.u1 * self.u2) as f32);
            let mut arg = q;
            arg.scale(self.d);
            coeff * arg.cosh()
        };

        let fourth_term = || -> DualVec<f32, f32, 1> {
            let mut coeff = q;
            coeff.scale((self.u1 + self.u2) as f32);
            let mut arg = q;
            arg.scale(self.d);
            coeff * arg.sinh()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }
}

fn plot_graph(roots: Vec<Vec<(f32, f32)>>, lower: f32, upper: f32) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("./plots/0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("V1=V2=5; v=0.6; w=1-v", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-std::f32::consts::PI..std::f32::consts::PI, lower..upper)?;

    chart.configure_mesh().draw()?;

    for series in roots {
        chart
            .draw_series(
                series.iter().map(|point| Circle::new(*point, 2, &RED))
            )?;
        }
    
    // chart
    //     .label("y = x^2")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED))

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn main() {
    let kp = KronigPenney::new(-10.0, -10.0, 1.0, 0.4, -10.5, -9.5, 1000, 1000, 0.0001);
    let roots = kp.find_roots();
    plot_graph(roots, kp.lower, kp.upper).unwrap();
}