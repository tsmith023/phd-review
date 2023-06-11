use std;
use num_dual::*;
use console::{Emoji};
use nalgebra::{ComplexField,DMatrix};
use indicatif::{ProgressBar};
use plotters::prelude::*;
use rayon::prelude::*;
use rusty_rootsearch::{root_search, RootSearchOptions};

const PI: f32 = std::f32::consts::PI;

#[derive(PartialEq)]
enum SystemType {
    Bulk,
    Finite,
}

struct KronigPenney {
    u1: f32,
    u2: f32,
    d: f32,
    v: f32,
    // w: f32,
    x1: f32,
    x2: f32,
    n: u32,
    lower: f32,
    upper: f32,
    resolution: i32,
    patience: i32,
    tolerance: f32,
}

struct RootsPerIterable {
    iterable: f32,
    roots: Vec<f32>,
}

trait RootsFinder {
    fn find_roots(&self, system: SystemType) -> Vec<RootsPerIterable>;
}

static LOOKING_GLASS: Emoji<'_, '_> = Emoji("🔍  ", "");

struct FindRootsInParallelOptions{
    lower: f32,
    upper: f32,
    resolution: i32,
    patience: i32,
    tolerance: f32,
}

fn find_bulk_roots_in_parallel<'a, F>(opts: FindRootsInParallelOptions, transcendental_equation: F) -> Vec<RootsPerIterable>
where F: Fn(Dual32, f32) -> Dual32 + Copy + Send + Sync + 'a {
    println!(
        "{} Finding roots in parallel...",
        LOOKING_GLASS
    );
    let pb = ProgressBar::new(opts.resolution as u64);

    let mut roots: Vec<RootsPerIterable> = Vec::new();
    let mut k_values: Vec<f32> = Vec::new();
    for i in 0..opts.resolution {
        let step = 2.0 * i as f32 / (opts.resolution as f32);
        let k = PI * (-1.0 + step);
        k_values.push(k);
    }
    k_values
        .par_iter()
        .map(|k| {
            pb.inc(1);
            let root_equation = move |q: Dual32| transcendental_equation(q, *k);
            let result = root_search::<_,Dual32,f32>(root_equation, RootSearchOptions{
                lower: opts.lower,
                upper: opts.upper,
                patience: opts.patience,
                tolerance: opts.tolerance,
                resolution: opts.resolution
            });
            RootsPerIterable{
                iterable: *k,
                roots: result.roots,
            }
        })
        .collect_into_vec(&mut roots);
    roots.sort_by(|a, b| a.iterable.partial_cmp(&b.iterable).unwrap());
    roots
}


impl RootsFinder for KronigPenney {
    fn find_roots(&self, system: SystemType) -> Vec<RootsPerIterable> {
        let opts = FindRootsInParallelOptions{
            lower: self.lower,
            upper: self.upper,
            resolution: self.resolution,
            patience: self.patience,
            tolerance: self.tolerance,
        };
        if self.lower < 0.0 && self.upper <= 0.0 {
            if self.u1 > 0.0 && self.u2 > 0.0 {
                panic!("No roots for negative energy and positive potential")
            }
            if system == SystemType::Bulk {
                return find_bulk_roots_in_parallel(opts, |q: Dual32, k: f32| self.transcendental_equation_for_negative_energy_bulk_roots(q, k));
            } else {
                // return find_roots_in_parallel(opts, |q: Dual32, k: f32| self.transcendental_equation_for_negative_energy_finite_roots(q, self.n));
                panic!("Finite system not implemented yet for negative energy roots")
            };
        } 
        else if self.lower >= 0.0 && self.upper > 0.0 {
            if self.u1 < 0.0 && self.u2 < 0.0 {
                panic!("No roots for positive energy and negative potential")
            }
            if system == SystemType::Bulk {
                return find_bulk_roots_in_parallel(opts, |q: Dual32, k: f32| self.transcendental_equation_for_positive_energy_bulk_roots(q, k))
            } else {
                panic!("Finite system not implemented yet for positive energy roots")
            }
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
        n: u32,
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
            x1,
            x2,
            n,
            lower,
            upper,
            resolution,
            patience,
            tolerance,
        }
    }

    fn transcendental_equation_for_positive_energy_bulk_roots(&self, q: Dual32, k: f32) -> Dual32 {
        let first_term = || -> Dual32 {
            let arg: Dual32 = k.into();
            arg.scale(self.d.into());
            let coeff = q.powi(2);
            coeff.scale(arg.cos());
            coeff
        };

        let second_term = || -> Dual32 {
            let diff = self.x2 - self.x1;
            diff.scale(2.0);

            let mut arg = q;
            let coeff1 = self.d - (2.0 * self.v);
            arg.scale(coeff1.into());

            let coeff2 = self.u1 * self.u2;
            arg = arg.cos();
            arg.scale(coeff2.into());

            arg
        };

        let third_term = || -> Dual32 {
            let coeff = q.powi(2) - Dual32::from_re((self.u1 * self.u2) as f32);
            let arg = q;
            arg.scale(self.d.into());
            coeff * arg.cos()
        };

        let fourth_term = || -> Dual32 {
            let coeff = q;
            coeff.scale((self.u1 + self.u2).into());
            let arg = q;
            arg.scale(self.d.into());
            coeff * arg.sin()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }

    fn transcendental_equation_for_negative_energy_bulk_roots(&self, q: Dual32, k: f32) -> Dual32 {
        let first_term = || -> Dual32 {
            q.powi(2) * (k * self.d).cos()
        };

        let second_term = || -> Dual32 {
            (q.powi(2) + (self.u1 * self.u2)) * (q * self.d).cosh()
        };

        let third_term = || -> Dual32 {
            q * (self.u1 + self.u2) * (q * self.d).sinh()
        };

        let fourth_term = || -> Dual32 {
            -Dual32::from_re(self.u1 * self.u2) * (q * (self.d - (2.0 * self.v))).cosh()
        };

        first_term() - second_term() - third_term() - fourth_term()
    }

    fn transcendental_equation_for_negative_energy_finite_roots(&self, q: Dual32, n: u32) -> Dual32 {
        let max = 2 * n as usize;
        let length = (n as f32) * self.d / 2.0;
        let x = |mut i: u32| -> f32 {
            i = i / 2;
            if n % 2 == 1 && i % 2 == 1 {
                ((-1 + 2 * (i as i32 - 1) / (n as i32 + 1)) as f32) * length / 2.0 + self.v
            }
            else if n % 2 == 1 && i % 2 == 0 {
                (-1 + 2 * i as i32 / (n as i32 + 1)) as f32 * length / 2.0
            }
            else if n % 2 == 0 && i % 2 == 0 {
                ((-1 + 2 * (i as i32 - 1) / n as i32) as f32) * length / 2.0 + (1 - (i - 1) / n) as f32 * self.v
            }
            else if n % 2 == 0 && i % 2 == 1 {
                (-1 + 2 * i as i32 / n as i32) as f32 * length / 2.0 - (i / n) as f32 * self.v
            }
            else {
                panic!("Something went wrong in the calculation of x.");
            }
        };
        let u = |mut i: u32| -> f32 {
            i = i / 2;
            if i % 2 == 0 {
                self.u1
            }
            else {
                self.u2
            }
        };
        let determine_element = |i: usize, j: usize| {
            let arg = q;

            let exp_expression = |exp_sign: i8| {
                let coeff = exp_sign as f32 * x(i as u32);
                arg.scale(coeff.into());
                arg.exp()
            };

            let chi_expression = |chi_sign: i8, exp_sign: i8| {
                let exp = exp_expression(exp_sign);

                let first = exp;
                let coeff1 = u(i as u32);
                first.scale(coeff1.into());

                let second = exp * q;
                let coeff2 = chi_sign as f32;
                second.scale(coeff2.into());

                first + second
            };

            match i % 2 {
                0 => {
                    if i == 0 {
                        if j == 0 {
                            return exp_expression(-1);
                        }
                        else if j == 1 {
                            return exp_expression(1);
                        }
                        else {
                            return Dual32::from_re(0.0);
                        }
                    }
                    else {
                        if j == i - 2 {
                            return chi_expression(-1, -1)
                        }
                        else if j == i - 1 {
                            return chi_expression(1, 1)
                        }
                        else if j == i {
                            return chi_expression(1, -1)
                        }
                        else if j == i + 1 {
                            return chi_expression(-1, 1)
                        }
                        else {
                            return Dual32::from_re(0.0);
                        }
                    }
                }
                1 => {
                    if i == max {
                        if j == max - 1 {
                            return exp_expression(-1);
                        }
                        else if j == max {
                            return exp_expression(1);
                        }
                        else {
                            return Dual32::from_re(0.0);
                        }
                    }
                    else {
                        if j == i - 2 {
                            return exp_expression(-1);
                        }
                        else if j == i - 1 {
                            return exp_expression(1);
                        }
                        else if j == i {
                            return exp_expression(-1);
                        }
                        else if j == i + 1 {
                            return exp_expression(1);
                        }
                        else {
                            return Dual32::from_re(0.0);
                        }
                    }
                }
                _ => {
                    panic!("Something went wrong in the calculation of the matrix.")
                }
            };
        };

        let matrix: DMatrix<Dual32> = DMatrix::from_fn(n as usize, n as usize, determine_element);

        let _svd = matrix.svd(false, false);

        return Dual32::from_re(0.0);
    }
}

fn plot_graph(results: Vec<RootsPerIterable>, lower: f32, upper: f32, path: String, caption: String) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-PI..PI, lower..upper)?;

    chart.configure_mesh().draw()?;

    for series in results {
        chart
            .draw_series(
                series.roots.iter().map(|root| Circle::new((series.iterable, *root), 2, &RED))
            )?;
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

fn main() {
    let kp = KronigPenney::new(-5.0, -5.0, 1.0, 0.45,100, -10.0, 0.0, 10000, 10000, 0.00001);
    let results = kp.find_roots(SystemType::Bulk);
    let caption = format!("u1={};u2={};v={};w={}", kp.u1, kp.u2, kp.v, kp.d - kp.v);
    let path = format!("./plots/{}.png", caption);
    plot_graph(results, kp.lower, kp.upper, path, caption).unwrap();
}